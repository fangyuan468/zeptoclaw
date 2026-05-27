//! Gateway command handler (multi-channel bot server).

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::FutureExt;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use zeptoclaw::agent::agui_events;
use zeptoclaw::bus::message::{
    OutboundMessageKind, OUTBOUND_CUSTOM_NAME_KEY, OUTBOUND_CUSTOM_PAYLOAD_KEY,
    OUTBOUND_CUSTOM_SUMMARY_KEY,
};
use zeptoclaw::bus::{InboundInterceptor, MessageBus, OutboundMessage};
use zeptoclaw::channels::{register_configured_channels, ChannelManager};
use zeptoclaw::config::watcher::ConfigWatcher;
use zeptoclaw::config::{Config, ContainerAgentBackend};
use zeptoclaw::health::{
    health_port, start_health_server, start_health_server_legacy, start_periodic_usage_flush,
    HealthRegistry, UsageMetrics,
};
use zeptoclaw::heartbeat::{ensure_heartbeat_file, HeartbeatService};
use zeptoclaw::providers::{
    configured_provider_names, resolve_runtime_provider, RUNTIME_SUPPORTED_PROVIDERS,
};
use zeptoclaw::tools::approval::{
    ApprovalRequest, ApprovalResponse, DEFAULT_APPROVAL_TIMEOUT_SECS,
};
use zeptoclaw::tools::approval_broker::{ApprovalBroker, ThreadKey};
use zeptoclaw::tools::thread_approval::{ThreadApprovalMode, ThreadApprovalStore};
use zeptoclaw::tools::thread_identity::ThreadIdentity;

use super::common::create_agent;
use super::heartbeat::heartbeat_file_path;

/// Start multi-channel gateway.
pub(crate) async fn cmd_gateway(
    containerized_flag: Option<String>,
    tunnel_flag: Option<String>,
) -> Result<()> {
    println!("Starting ZeptoClaw Gateway...");

    // Load configuration
    let mut config = Config::load().with_context(|| "Failed to load configuration")?;

    // Startup guard — check for consecutive crash degradation
    let guard = if config.gateway.startup_guard.enabled {
        let g = zeptoclaw::StartupGuard::new(
            config.gateway.startup_guard.crash_threshold,
            config.gateway.startup_guard.window_secs,
        );
        match g.check() {
            Ok(true) => {
                warn!(
                    threshold = config.gateway.startup_guard.crash_threshold,
                    "Startup guard: consecutive crashes detected — entering degraded mode"
                );
                warn!("Degraded mode: shell and filesystem write tools disabled");
                config.tools.deny = vec![
                    "shell".to_string(),
                    "write_file".to_string(),
                    "edit_file".to_string(),
                ];
            }
            Ok(false) => {}
            Err(e) => warn!("Startup guard check failed (continuing normally): {}", e),
        }
        Some(g)
    } else {
        None
    };

    // --containerized [docker|apple] overrides config backend
    let containerized = containerized_flag.is_some();
    if let Some(ref b) = containerized_flag {
        if b != "auto" {
            config.container_agent.backend = match b.to_lowercase().as_str() {
                "docker" => ContainerAgentBackend::Docker,
                #[cfg(target_os = "macos")]
                "apple" => ContainerAgentBackend::Apple,
                "auto" => ContainerAgentBackend::Auto,
                other => {
                    #[cfg(target_os = "macos")]
                    return Err(anyhow::anyhow!(
                        "Unknown backend '{}'. Use: docker or apple",
                        other
                    ));
                    #[cfg(not(target_os = "macos"))]
                    return Err(anyhow::anyhow!("Unknown backend '{}'. Use: docker", other));
                }
            };
        }
    }

    // Start tunnel if requested
    let mut _tunnel: Option<Box<dyn zeptoclaw::tunnel::TunnelProvider>> = None;
    let tunnel_provider = tunnel_flag.or(config.tunnel.provider.clone());
    if let Some(ref provider) = tunnel_provider {
        let mut tunnel_config = config.tunnel.clone();
        tunnel_config.provider = Some(provider.clone());
        let mut t = zeptoclaw::tunnel::create_tunnel(&tunnel_config)
            .with_context(|| format!("Failed to create {} tunnel", provider))?;

        let gateway_port = config.gateway.port;
        let tunnel_url = t
            .start(gateway_port)
            .await
            .with_context(|| format!("Failed to start {} tunnel", provider))?;

        println!("Tunnel active: {}", tunnel_url);
        _tunnel = Some(t);
    }

    // Create message bus
    let bus = Arc::new(MessageBus::new());

    // Create usage metrics tracker
    let metrics = Arc::new(UsageMetrics::new());

    // Start legacy health check server (liveness + readiness via UsageMetrics)
    let hp = health_port();
    let health_handle = match start_health_server_legacy(hp, Arc::clone(&metrics)).await {
        Ok(handle) => {
            info!(
                port = hp,
                "Health endpoints available at /healthz and /readyz"
            );
            Some(handle)
        }
        Err(e) => {
            warn!(error = %e, "Failed to start health server (non-fatal)");
            None
        }
    };

    // Create HealthRegistry (shared between health server and channel supervisor)
    let health_registry = HealthRegistry::new();
    health_registry.set_metrics(Arc::clone(&metrics));

    // Start HealthRegistry-based server if config.health.enabled
    if config.health.enabled {
        let registry = health_registry.clone();
        let host = config.health.host.clone();
        let port = config.health.port;
        tokio::spawn(async move {
            match start_health_server(&host, port, registry).await {
                Ok(handle) => {
                    info!(
                        host = %host,
                        port = port,
                        "Named-check health server listening on /health and /ready"
                    );
                    let _ = handle.await;
                }
                Err(e) => {
                    tracing::error!("Named-check health server error: {}", e);
                }
            }
        });
        info!(
            "Health server enabled on {}:{}",
            config.health.host, config.health.port
        );
    }

    // Create shutdown watch channel for periodic usage flush
    let (usage_shutdown_tx, usage_shutdown_rx) = tokio::sync::watch::channel(false);
    let usage_flush_handle = start_periodic_usage_flush(Arc::clone(&metrics), usage_shutdown_rx);

    // Determine agent backend: containerized or in-process
    let mut proxy = None;
    let proxy_handle = if containerized {
        info!("Starting gateway with containerized agent mode");

        // Resolve backend (auto-detect or explicit from config)
        let backend = zeptoclaw::gateway::resolve_backend(&config.container_agent)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        info!("Resolved container backend: {}", backend);

        // Validate the resolved backend
        match backend {
            zeptoclaw::gateway::ResolvedBackend::Docker => {
                validate_docker_available(configured_docker_binary(&config.container_agent))
                    .await?;
            }
            #[cfg(target_os = "macos")]
            zeptoclaw::gateway::ResolvedBackend::Apple => {
                validate_apple_available().await?;
            }
        }

        // Check image exists (Docker-specific)
        let image = &config.container_agent.image;
        if backend == zeptoclaw::gateway::ResolvedBackend::Docker {
            let docker_binary = configured_docker_binary(&config.container_agent);
            let image_check = tokio::process::Command::new(docker_binary)
                .args(["image", "inspect", image])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await;

            if !image_check.map(|s| s.success()).unwrap_or(false) {
                eprintln!(
                    "Warning: Docker image '{}' not found (checked via '{}').",
                    image, docker_binary
                );
                eprintln!("Build it with: {} build -t {} .", docker_binary, image);
                return Err(anyhow::anyhow!(
                    "Docker image '{}' not found (checked via '{}')",
                    image,
                    docker_binary
                ));
            }
        }

        info!("Using container image: {} (backend={})", image, backend);

        let proxy_instance = Arc::new(zeptoclaw::gateway::ContainerAgentProxy::new(
            config.clone(),
            bus.clone(),
            backend,
        ));
        proxy_instance.set_usage_metrics(Arc::clone(&metrics));
        let proxy_for_task = Arc::clone(&proxy_instance);
        let proxy_metrics = Arc::clone(&metrics);
        let proxy_guard = guard.clone();
        proxy = Some(proxy_instance);

        Some(tokio::spawn(async move {
            let result = proxy_for_task.start().await;
            proxy_metrics.set_ready(false);
            match result {
                Err(e) => {
                    error!("Container agent proxy error: {}", e);
                    if let Some(ref g) = proxy_guard {
                        if let Err(re) = g.record_crash() {
                            warn!("Failed to record crash: {}", re);
                        }
                    }
                }
                Ok(()) => warn!("Container agent proxy stopped"),
            }
        }))
    } else {
        // Validate provider for in-process mode
        let runtime_provider_name = resolve_runtime_provider(&config).map(|provider| provider.name);
        if runtime_provider_name.is_none() {
            let configured = configured_provider_names(&config);
            if configured.is_empty() {
                error!("No AI provider configured. Set ZEPTOCLAW_PROVIDERS_ANTHROPIC_API_KEY");
                error!("or add your API key to {:?}", Config::path());
            } else {
                error!(
                    "Configured provider(s) are not supported by this runtime: {}",
                    configured.join(", ")
                );
                error!(
                    "Currently supported runtime providers: {}",
                    RUNTIME_SUPPORTED_PROVIDERS.join(", ")
                );
            }
            std::process::exit(1);
        }
        None
    };

    // Preflight: validate model-provider compatibility before starting anything.
    if !containerized {
        let model_diags = zeptoclaw::config::validate::validate_model_provider_compat(&config);
        for diag in &model_diags {
            match diag.level {
                zeptoclaw::config::validate::DiagnosticLevel::Error => {
                    error!("{}", diag);
                }
                zeptoclaw::config::validate::DiagnosticLevel::Warn => {
                    warn!("{}", diag);
                }
                zeptoclaw::config::validate::DiagnosticLevel::Ok => {
                    info!("{}", diag);
                }
            }
        }
        let has_errors = model_diags
            .iter()
            .any(|d| d.level == zeptoclaw::config::validate::DiagnosticLevel::Error);
        if has_errors {
            eprintln!();
            eprintln!("ERROR: Model-provider mismatch detected.");
            eprintln!(
                "  Fix your config ({:?}) or run 'zeptoclaw onboard'.",
                Config::path()
            );
            eprintln!("  Run 'zeptoclaw config check' for details.");
            eprintln!();
            std::process::exit(1);
        }
    }

    // Create in-process agent (only needed when not containerized)
    let mut agent = if !containerized {
        let agent = create_agent(config.clone(), bus.clone()).await?;
        agent.set_usage_metrics(Arc::clone(&metrics)).await;
        Some(agent)
    } else {
        None
    };

    // --- Tool-approval handler (generic, works for all channels) ----------
    let broker = Arc::new(ApprovalBroker::new());

    // Process-level `(user, agent)` identity stamped on the host by the
    // scheduler (`thread_meta.json`). Falls back to `Unknown` outside
    // the containerised path so dev / in-process callers behave like
    // pre-PR1 (chat_id-keyed broker buckets).
    let thread_identity = {
        let cfg_path = Config::path();
        let cfg_dir = cfg_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        Arc::new(ThreadIdentity::load_from_config_dir(&cfg_dir))
    };
    if let Some(key) = thread_identity.as_known() {
        info!(
            user = %key.user,
            agent = %key.agent,
            "Loaded thread identity for HITL approval routing"
        );
    } else {
        info!(
            "thread_meta.json absent — approval broker will fall back to chat_id-keyed \
             buckets (legacy behaviour, no AutoApprove persistence across threads)."
        );
    }

    // Per-thread approval mode store, persisted at
    // `<config_dir>/approval_thread.json`. Missing/corrupt file → empty
    // store (all threads default to RequireApproval, identical to the
    // pre-PR2 baseline). Same parent dir as `thread_meta.json`.
    let approval_store = {
        let cfg_path = Config::path();
        let cfg_dir = cfg_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        Arc::new(ThreadApprovalStore::open(cfg_dir))
    };

    // In-memory pending `/skip-approval` confirmation slots. Keyed by
    // `ThreadKey` so a confirm reply on the same thread (regardless of
    // channel) can finalise the switch. Cleared on success, on any
    // non-`confirm` follow-up, or after the 60s deadline.
    let pending_confirms: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
        Arc::new(StdMutex::new(HashMap::new()));

    // Inbound interceptor — three orthogonal jobs, evaluated in this
    // strict order to keep semantics predictable:
    //
    //   1. resolve a pending `/skip-approval` confirmation (so
    //      "confirm" works even if the user later starts typing other
    //      commands on the same line);
    //   2. parse thread approval commands and run them (consumes the
    //      message — the agent never sees a `/skip-approval` line);
    //   3. fall back to single-request `yes / no` approval-reply
    //      routing. Batch `yes all` / `no all` was intentionally
    //      dropped — every pending approval is now decided one at a
    //      time, with the per-thread `/skip-approval` toggle covering
    //      the "auto-approve everything for this thread" case.
    bus.set_inbound_interceptor({
        let broker = Arc::clone(&broker);
        let identity = Arc::clone(&thread_identity);
        let store = Arc::clone(&approval_store);
        let pending = Arc::clone(&pending_confirms);
        let bus_for_replies = Arc::clone(&bus);
        Arc::new(move |msg: &zeptoclaw::bus::InboundMessage| {
            let thread = identity.resolve_key(&msg.chat_id);
            let trimmed = msg.content.trim();

            // Step 1: pending confirm (if any) takes priority. A
            // matching `confirm` finalises the switch; any other text
            // silently cancels the pending slot (and we keep
            // processing so the user's actual message still gets a
            // chance to be parsed as a command or routed to the
            // agent).
            let confirm_result =
                handle_pending_confirm(&pending, &thread, trimmed, msg, &bus_for_replies, &store);
            if matches!(confirm_result, ConfirmOutcome::Consumed) {
                return true;
            }

            // Step 2: thread approval commands.
            if let Some(cmd) = parse_thread_command(trimmed) {
                handle_thread_command(
                    cmd,
                    &thread,
                    msg,
                    &store,
                    &broker,
                    &pending,
                    &bus_for_replies,
                );
                return true;
            }

            // Step 3: yes/no approval reply routing.
            let Some(approved) = parse_approval_reply(&msg.content) else {
                return false;
            };
            match msg.metadata.get("inject_kind").map(String::as_str) {
                Some("approval_response") => {
                    let Some(request_id) = msg.metadata.get("request_id").map(String::as_str)
                    else {
                        return false;
                    };
                    if request_id.is_empty() {
                        broker.resolve_fifo(&thread, approved)
                    } else {
                        broker.resolve_by_id(&thread, request_id, approved)
                    }
                }
                Some(_) => false,
                None => broker.resolve_fifo(&thread, approved),
            }
        }) as InboundInterceptor
    });

    // Background GC: drop pending broker entries whose handler future
    // was cancelled out-of-band (e.g. SSE stream torn down before the
    // normal 120s timeout fires). Without this, long-lived gateways
    // accumulate dead `oneshot::Sender`s indefinitely.
    spawn_broker_sweep(Arc::clone(&broker));

    let approval_timeout = resolve_approval_timeout(&config);
    if let Some(ref agent) = agent {
        agent.set_thread_identity(Arc::clone(&thread_identity));
        register_approval_handler(
            agent,
            &broker,
            &bus,
            approval_timeout,
            &thread_identity,
            &approval_store,
        )
        .await;
    }

    // Create channel manager with health supervision
    let mut channel_manager = ChannelManager::new(bus.clone(), config.clone());
    channel_manager.set_health_registry(health_registry.clone());

    // Register channels via factory.
    let channel_count = register_configured_channels(&channel_manager, bus.clone(), &config).await;
    if channel_count == 0 {
        warn!(
            "No channels configured. Enable channels in {:?}",
            Config::path()
        );
        warn!("The agent loop will still run but won't receive messages from external sources.");
    } else {
        info!("Registered {} channel(s)", channel_count);
    }

    // Start all channels
    channel_manager
        .start_all()
        .await
        .with_context(|| "Failed to start channels")?;

    let heartbeat_service = if config.heartbeat.enabled {
        let hb_path = heartbeat_file_path(&config);
        match ensure_heartbeat_file(&hb_path).await {
            Ok(true) => info!("Created heartbeat file template at {:?}", hb_path),
            Ok(false) => {}
            Err(e) => warn!("Failed to initialize heartbeat file {:?}: {}", hb_path, e),
        }

        let (hb_channel, hb_chat_id) = config
            .heartbeat
            .deliver_to
            .as_deref()
            .and_then(|s| {
                let parsed = parse_deliver_to(s);
                if parsed.is_none() {
                    warn!(
                        "heartbeat.deliver_to {:?} is not in 'channel:chat_id' format; \
                         falling back to pseudo-channel",
                        s
                    );
                }
                parsed
            })
            .unwrap_or_else(|| ("heartbeat".to_string(), "system".to_string()));

        let service = Arc::new(HeartbeatService::new(
            hb_path,
            config.heartbeat.interval_secs,
            bus.clone(),
            &hb_channel,
            &hb_chat_id,
        ));
        service.start().await?;
        Some(service)
    } else {
        None
    };

    // Start memory hygiene scheduler
    let _hygiene_handle = match zeptoclaw::memory::longterm::LongTermMemory::new() {
        Ok(ltm) => {
            let ltm = Arc::new(tokio::sync::Mutex::new(ltm));
            Some(zeptoclaw::memory::hygiene::start_hygiene_scheduler(
                ltm,
                config.memory.hygiene.clone(),
            ))
        }
        Err(e) => {
            warn!("Memory hygiene scheduler not started: {}", e);
            None
        }
    };

    // Start device service if configured
    // TODO: publish to MessageBus for channel delivery once InboundMessage wrapping is settled
    let _device_handle =
        zeptoclaw::devices::DeviceService::new(config.devices.enabled, config.devices.monitor_usb)
            .start()
            .map(|mut rx| {
                tokio::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        tracing::info!("Device event: {}", event.format_message());
                    }
                })
            });

    // Start agent loop in background (only for in-process mode)
    let mut agent_handle = if let Some(ref agent) = agent {
        let agent_clone = Arc::clone(agent);
        let agent_metrics = Arc::clone(&metrics);
        let agent_guard = guard.clone();
        Some(tokio::spawn(async move {
            let result = agent_clone.start().await;
            agent_metrics.set_ready(false);
            match result {
                Err(e) => {
                    error!("Agent loop error: {}", e);
                    if let Some(ref g) = agent_guard {
                        if let Err(re) = g.record_crash() {
                            warn!("Failed to record crash: {}", re);
                        }
                    }
                }
                Ok(()) => warn!("Agent loop stopped"),
            }
        }))
    } else {
        None
    };

    // Mark gateway as ready for /readyz
    metrics.set_ready(true);

    // Record clean start (reset crash counter)
    if let Some(ref g) = guard {
        if let Err(e) = g.record_clean_start() {
            warn!("Failed to record clean start: {}", e);
        }
    }

    println!();
    if !config.tools.deny.is_empty() {
        println!("  WARNING: DEGRADED MODE — dangerous tools disabled after consecutive crashes");
        println!("  Fix the underlying issue and restart to clear.");
        println!();
    }
    if containerized {
        println!("Gateway is running (containerized mode). Press Ctrl+C to stop.");
    } else {
        println!("Gateway is running. Press Ctrl+C to stop.");
    }
    println!();

    // Config watcher (30s polling) for hot-reload.
    let (reload_tx, mut reload_rx) = mpsc::unbounded_channel::<Config>();
    let (reload_shutdown_tx, reload_shutdown_rx) = watch::channel(false);
    let watcher_handle = tokio::spawn(
        ConfigWatcher::default_path(Duration::from_secs(30)).watch(reload_tx, reload_shutdown_rx),
    );

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                break;
            }
            maybe_cfg = reload_rx.recv() => {
                let Some(new_config) = maybe_cfg else {
                    break;
                };

                let changed_sections = diff_hot_reload_sections(&config, &new_config);
                if changed_sections.is_empty() {
                    continue;
                }

                info!(
                    sections = %changed_sections.join(", "),
                    "Applying hot-reloaded config sections"
                );

                let old_config = config.clone();
                config = new_config;

                // Rebuild in-process agent to apply provider + safety changes.
                if !containerized {
                    if let Some(ref running_agent) = agent {
                        running_agent.stop();
                        running_agent.shutdown_mcp_clients().await;
                    }
                    if let Some(handle) = agent_handle.take() {
                        let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
                    }

                    match create_agent(config.clone(), bus.clone()).await {
                        Ok(new_agent) => {
                            new_agent.set_usage_metrics(Arc::clone(&metrics)).await;
                            new_agent.set_thread_identity(Arc::clone(&thread_identity));
                            register_approval_handler(
                                &new_agent,
                                &broker,
                                &bus,
                                approval_timeout,
                                &thread_identity,
                                &approval_store,
                            )
                            .await;
                            let agent_clone = Arc::clone(&new_agent);
                            let agent_metrics = Arc::clone(&metrics);
                            let agent_guard = guard.clone();
                            agent_handle = Some(tokio::spawn(async move {
                                let result = agent_clone.start().await;
                                agent_metrics.set_ready(false);
                                match result {
                                    Err(e) => {
                                        error!("Agent loop error: {}", e);
                                        if let Some(ref g) = agent_guard {
                                            if let Err(re) = g.record_crash() {
                                                warn!("Failed to record crash: {}", re);
                                            }
                                        }
                                    }
                                    Ok(()) => warn!("Agent loop stopped"),
                                }
                            }));
                            agent = Some(new_agent);
                        }
                        Err(e) => {
                            config = old_config;
                            warn!("Hot-reload failed to rebuild agent, keeping prior config: {}", e);
                            continue;
                        }
                    }
                } else {
                    warn!("Config hot-reload for containerized mode is not yet supported");
                }

                // Rebuild channels only if channel config actually changed.
                if changed_sections.contains(&"channels") {
                    if let Err(e) = channel_manager.stop_all().await {
                        warn!("Failed to stop channels during hot-reload: {}", e);
                    }
                    let mut new_manager = ChannelManager::new(bus.clone(), config.clone());
                    new_manager.set_health_registry(health_registry.clone());
                    let count = register_configured_channels(&new_manager, bus.clone(), &config).await;
                    if count == 0 {
                        warn!("No channels configured after hot-reload");
                    }
                    if let Err(e) = new_manager.start_all().await {
                        config = old_config;
                        warn!("Failed to start channels after hot-reload, keeping previous config: {}", e);
                        continue;
                    }
                    channel_manager = new_manager;
                }
            }
        }
    }

    println!();
    println!("Shutting down...");

    // Mark not ready immediately
    metrics.set_ready(false);

    // Signal usage flush to emit final summary
    let _ = usage_shutdown_tx.send(true);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), usage_flush_handle).await;

    if let Some(service) = &heartbeat_service {
        service.stop().await;
    }

    // Stop agent or proxy
    if let Some(ref agent) = agent {
        agent.stop();
        agent.shutdown_mcp_clients().await;
    }
    if let Some(ref proxy) = proxy {
        proxy.stop();
    }

    // Stop all channels
    channel_manager
        .stop_all()
        .await
        .with_context(|| "Failed to stop channels")?;

    // Stop config watcher
    let _ = reload_shutdown_tx.send(true);
    let _ = tokio::time::timeout(Duration::from_secs(2), watcher_handle).await;

    // Wait for agent/proxy to stop
    if let Some(handle) = agent_handle {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
    }
    if let Some(handle) = proxy_handle {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
    }

    // Stop health server
    if let Some(handle) = health_handle {
        handle.abort();
    }

    println!("Gateway stopped.");
    Ok(())
}

/// Validate that Docker is available.
async fn validate_docker_available(docker_binary: &str) -> Result<()> {
    if !zeptoclaw::gateway::is_docker_available_with_binary(docker_binary).await {
        return Err(anyhow::anyhow!(
            "Docker is not available via '{}'. Install Docker or run without --containerized.",
            docker_binary
        ));
    }
    Ok(())
}

fn configured_docker_binary(config: &zeptoclaw::config::ContainerAgentConfig) -> &str {
    config
        .docker_binary
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("docker")
}

/// Validate that Apple Container is available (macOS only).
#[cfg(target_os = "macos")]
async fn validate_apple_available() -> Result<()> {
    if !zeptoclaw::gateway::is_apple_container_available().await {
        return Err(anyhow::anyhow!(
            "Apple Container is not available. Requires macOS 15+ with `container` CLI installed."
        ));
    }
    Ok(())
}

/// Parse a `deliver_to` string in `"channel:chat_id"` format.
/// Returns `None` if the string is missing a colon or either part is empty.
fn parse_deliver_to(s: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

fn diff_hot_reload_sections(old: &Config, new: &Config) -> Vec<&'static str> {
    let mut changed = Vec::new();
    if section_changed(&old.providers, &new.providers) {
        changed.push("providers");
    }
    if section_changed(&old.channels, &new.channels) {
        changed.push("channels");
    }
    if section_changed(&old.safety, &new.safety) {
        changed.push("safety");
    }
    if section_changed(&old.agents, &new.agents) {
        changed.push("agents");
    }
    changed
}

fn section_changed<T: serde::Serialize>(old: &T, new: &T) -> bool {
    serde_json::to_value(old).ok() != serde_json::to_value(new).ok()
}

fn resolve_approval_timeout(config: &Config) -> u64 {
    let t = config.approval.approval_timeout_secs;
    if t > 0 {
        t
    } else {
        DEFAULT_APPROVAL_TIMEOUT_SECS
    }
}

/// Background interval ticker that drops `ApprovalBroker` entries whose
/// `oneshot::Sender` got orphaned (handler future cancelled before
/// reaching its normal timeout). The sweep frequency / TTL are picked
/// conservatively so the failure mode is "an extra few minutes of
/// `RecvError` for the handler" rather than aggressive cleanup that
/// could race with a slow user reply.
fn spawn_broker_sweep(broker: Arc<ApprovalBroker>) {
    // Sweep every 30s. Anything older than 10 minutes is considered
    // abandoned — even the 120s `DEFAULT_APPROVAL_TIMEOUT_SECS` worst-
    // case + retry chatter fits well under that ceiling, while typical
    // entries (resolved within seconds) never trigger the sweep at all.
    const SWEEP_INTERVAL: Duration = Duration::from_secs(30);
    const ENTRY_TTL: Duration = Duration::from_secs(600);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Discard the immediate first tick — no entries can have aged
        // out yet at startup.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let swept = broker.sweep_expired(ENTRY_TTL);
            if swept > 0 {
                tracing::debug!(
                    swept,
                    ttl_secs = ENTRY_TTL.as_secs(),
                    "ApprovalBroker swept abandoned entries"
                );
            }
        }
    });
}

fn parse_approval_reply(content: &str) -> Option<bool> {
    match content.trim().to_lowercase().as_str() {
        "yes" | "y" | "approve" => Some(true),
        "no" | "n" | "deny" => Some(false),
        _ => None,
    }
}

fn is_acp_channel(channel: &str) -> bool {
    channel == "acp" || channel == "acp_http"
}

fn approval_decision_label(outcome: &ApprovalResponse) -> Option<&'static str> {
    match outcome {
        ApprovalResponse::Approved => Some("approve"),
        ApprovalResponse::Denied(_) => Some("deny"),
        ApprovalResponse::TimedOut => None,
    }
}

// --- Thread-level approval commands ---------------------------------------

/// User-issued thread-level command. Parsed by `parse_thread_command`
/// from the leading slash and acted on inside `set_inbound_interceptor`.
/// These are channel-agnostic — they work the same over Discord, the
/// Tauri client (via ACP/HTTP), and any other channel that routes
/// inbound messages through this interceptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThreadCommand {
    SkipApproval,
    RequireApproval,
    Status,
    Help,
}

/// Help text for thread-level commands. Owned here so the wording stays
/// in sync between the `/help` response and the inline mode hint on
/// approval cards. Kept as a single block: chat surfaces don't render
/// complex layouts uniformly.
const THREAD_HELP_TEXT: &str = "\
**zeptoclaw — available commands**\n\
\n\
Approval mode (per-thread, persistent):\n\
• `/skip-approval` — disable manual approval. Reply `confirm` within 60s to take effect.\n\
• `/require-approval` — re-enable manual approval (default).\n\
• `/approval-status` — show current mode + pending count.\n\
\n\
Misc:\n\
• `/help` — show this message.\n\
\n\
While an approval card is active, reply `yes` / `no` to that specific card.\n\
HardFloor rules (destructive shell commands) always require approval and \
cannot be skipped by `/skip-approval`.";

/// Recognise documented command aliases. Case-insensitive; trailing /
/// leading whitespace tolerated by the caller (`trim()` is done before
/// this is invoked).
fn parse_thread_command(content: &str) -> Option<ThreadCommand> {
    let lower = content.to_ascii_lowercase();
    match lower.as_str() {
        "/skip-approval" | "/skip_approval" => Some(ThreadCommand::SkipApproval),
        "/require-approval" | "/require_approval" => Some(ThreadCommand::RequireApproval),
        "/approval-status" | "/approval_status" => Some(ThreadCommand::Status),
        "/help" | "/?" => Some(ThreadCommand::Help),
        _ => None,
    }
}

/// In-memory record of a `/skip-approval` request awaiting the user's
/// `confirm` reply. Cleared after the user confirms, cancels (any other
/// text), or the deadline passes.
struct ConfirmPending {
    deadline: Instant,
    channel: String,
    chat_id: String,
}

/// How long the user has to reply `confirm` after `/skip-approval`.
const CONFIRM_WINDOW: Duration = Duration::from_secs(60);

/// Outcome of the pre-command "did this message confirm a pending
/// `/skip-approval`?" check. `Consumed` short-circuits the interceptor;
/// `Fallthrough` lets it run the rest of its pipeline (thread command
/// parsing, then approval reply parsing).
enum ConfirmOutcome {
    Consumed,
    Fallthrough,
}

/// Step 1 of the inbound interceptor. Returns `Consumed` only when the
/// message *was* a `confirm` for a live pending slot. Stale, mismatched,
/// or absent pending slots all return `Fallthrough` so the message can
/// be re-evaluated against the command/reply parsers — keeps the user's
/// other text from being swallowed.
fn handle_pending_confirm(
    pending: &Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>>,
    thread: &ThreadKey,
    trimmed: &str,
    msg: &zeptoclaw::bus::InboundMessage,
    bus: &Arc<MessageBus>,
    store: &Arc<ThreadApprovalStore>,
) -> ConfirmOutcome {
    let now = Instant::now();
    let entry = {
        let mut guard = pending.lock().expect("pending_confirms poisoned");
        match guard.get(thread) {
            Some(p) if p.deadline > now => Some(guard.remove(thread).unwrap()),
            Some(_) => {
                // Expired — silently drop and keep going.
                guard.remove(thread);
                None
            }
            None => None,
        }
    };
    let Some(entry) = entry else {
        return ConfirmOutcome::Fallthrough;
    };
    // Live pending slot present. Only an exact `confirm` token
    // commits; any other text cancels (drop the slot and let the
    // message fall through to the other parsers).
    if trimmed.eq_ignore_ascii_case("confirm") {
        let source = msg.channel.clone();
        match store.set(thread, ThreadApprovalMode::AutoApprove, &source) {
            Ok(_) => {
                tracing::info!(
                    target: "audit::approval",
                    user = %thread.user,
                    agent = %thread.agent,
                    decision = "mode_changed",
                    new_mode = "auto_approve",
                    source = %source,
                    "thread approval mode confirmed"
                );
                spawn_reply(
                    bus,
                    &entry.channel,
                    &entry.chat_id,
                    "✅ AutoApprove enabled for this thread. \
                     Use `/require-approval` to switch back.",
                );
            }
            Err(e) => {
                warn!("Failed to persist AutoApprove state: {}", e);
                spawn_reply(
                    bus,
                    &entry.channel,
                    &entry.chat_id,
                    "⚠️ Couldn't persist the change. Try again.",
                );
            }
        }
        ConfirmOutcome::Consumed
    } else {
        // Cancellation path: silently drop the pending slot so the
        // user's message can be parsed normally below. We don't
        // emit a "cancelled" notice to keep replies on track with
        // what the user actually typed.
        ConfirmOutcome::Fallthrough
    }
}

/// Step 2 of the inbound interceptor. Runs after step 1 has cleared or
/// preserved any pending confirm. Always consumes the message
/// (`return true` in the caller) so the user's command never reaches
/// the LLM as conversational input.
fn handle_thread_command(
    cmd: ThreadCommand,
    thread: &ThreadKey,
    msg: &zeptoclaw::bus::InboundMessage,
    store: &Arc<ThreadApprovalStore>,
    broker: &Arc<ApprovalBroker>,
    pending: &Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>>,
    bus: &Arc<MessageBus>,
) {
    match cmd {
        ThreadCommand::SkipApproval => {
            if matches!(store.get(thread).mode, ThreadApprovalMode::AutoApprove) {
                spawn_reply(
                    bus,
                    &msg.channel,
                    &msg.chat_id,
                    "ℹ️ AutoApprove is already active for this thread. \
                     Use `/require-approval` to switch back.",
                );
                return;
            }
            // Stash a pending confirm slot. Replaces any prior pending
            // for the same thread — last-write-wins keeps the UI
            // predictable.
            {
                let mut guard = pending.lock().expect("pending_confirms poisoned");
                guard.insert(
                    thread.clone(),
                    ConfirmPending {
                        deadline: Instant::now() + CONFIRM_WINDOW,
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                    },
                );
            }
            spawn_reply(
                bus,
                &msg.channel,
                &msg.chat_id,
                "⚠️ You're about to disable manual approval for this thread. \
                 Dangerous tools will run without prompting. \
                 Reply `confirm` within 60s to proceed (any other reply cancels).",
            );
        }
        ThreadCommand::RequireApproval => {
            // No confirmation needed — turning safety ON is always safe.
            {
                let mut guard = pending.lock().expect("pending_confirms poisoned");
                guard.remove(thread);
            }
            let source = msg.channel.clone();
            match store.set(thread, ThreadApprovalMode::RequireApproval, &source) {
                Ok(_) => {
                    tracing::info!(
                        target: "audit::approval",
                        user = %thread.user,
                        agent = %thread.agent,
                        decision = "mode_changed",
                        new_mode = "require_approval",
                        source = %source,
                        "thread approval mode restored"
                    );
                    spawn_reply(
                        bus,
                        &msg.channel,
                        &msg.chat_id,
                        "✅ Manual approval is back on for this thread.",
                    );
                }
                Err(e) => {
                    warn!("Failed to persist RequireApproval state: {}", e);
                    spawn_reply(
                        bus,
                        &msg.channel,
                        &msg.chat_id,
                        "⚠️ Couldn't persist the change. Try again.",
                    );
                }
            }
        }
        ThreadCommand::Status => {
            let state = store.get(thread);
            let pending_count = broker.pending_count(thread);
            let mode_label = match state.mode {
                ThreadApprovalMode::RequireApproval => "RequireApproval (default)",
                ThreadApprovalMode::AutoApprove => "AutoApprove",
            };
            let updated = if state.updated_by.is_empty() {
                "never changed".to_string()
            } else {
                format!(
                    "last changed by `{}` at {}",
                    state.updated_by,
                    state.updated_at.to_rfc3339()
                )
            };
            let pending_line = if pending_count == 0 {
                "no pending approvals".to_string()
            } else {
                format!("{} pending approval(s) on this thread", pending_count)
            };
            let body = format!(
                "**Approval status**\n• Mode: `{}`\n• {}\n• {}\n\n\
                 Send `/help` for the full command list.",
                mode_label, updated, pending_line,
            );
            spawn_reply(bus, &msg.channel, &msg.chat_id, &body);
        }
        ThreadCommand::Help => {
            spawn_reply(bus, &msg.channel, &msg.chat_id, THREAD_HELP_TEXT);
        }
    }
}

/// Fire-and-forget outbound publish. Used by the interceptor which is a
/// sync `Fn` — we can't await here, so spawn a task on the current
/// runtime. Errors are logged; the interceptor never blocks.
fn spawn_reply(bus: &Arc<MessageBus>, channel: &str, chat_id: &str, body: &str) {
    let bus = Arc::clone(bus);
    let channel = channel.to_string();
    let chat_id = chat_id.to_string();
    let body = body.to_string();
    tokio::spawn(async move {
        if let Err(e) = bus
            .publish_outbound(OutboundMessage::new(&channel, &chat_id, &body))
            .await
        {
            warn!("Failed to publish thread approval reply: {}", e);
        }
    });
}

async fn register_approval_handler(
    agent: &zeptoclaw::agent::AgentLoop,
    broker: &Arc<ApprovalBroker>,
    bus: &Arc<MessageBus>,
    timeout_secs: u64,
    thread_identity: &Arc<ThreadIdentity>,
    approval_store: &Arc<ThreadApprovalStore>,
) {
    let broker = Arc::clone(broker);
    let bus = Arc::clone(bus);
    let identity = Arc::clone(thread_identity);
    let store = Arc::clone(approval_store);
    agent
        .set_approval_handler(move |request: ApprovalRequest| {
            let broker = Arc::clone(&broker);
            let bus = Arc::clone(&bus);
            let identity = Arc::clone(&identity);
            let store = Arc::clone(&store);
            async move {
                let chat_id = match request.chat_id.as_deref() {
                    Some(id) if !id.is_empty() => id.to_string(),
                    _ => {
                        return ApprovalResponse::Denied(
                            "No chat routing context for approval".into(),
                        )
                    }
                };
                let channel = match request.channel.as_deref() {
                    Some(ch) if !ch.is_empty() => ch.to_string(),
                    _ => {
                        return ApprovalResponse::Denied(
                            "No channel routing context for approval".into(),
                        )
                    }
                };

                // Prefer the request-supplied `(user_id, agent_id)` (the
                // agent loop fills these from the same `ThreadIdentity`
                // we have here, but having the request carry them keeps
                // the handler decoupled from process state and lets a
                // future caller override). Fall back to the process
                // identity, then to chat_id-keyed buckets.
                let thread = match (request.user_id.as_deref(), request.agent_id.as_deref()) {
                    (Some(u), Some(a)) if !u.is_empty() && !a.is_empty() => ThreadKey::new(u, a),
                    _ => identity.resolve_key(&chat_id),
                };

                // PR2: thread-level AutoApprove bypass. HardFloor
                // (PR3) escalations are NEVER bypassable, so we
                // explicitly check `hard_floor_reason` first. Only the
                // intersection {AutoApprove ∧ ¬HardFloor} skips the
                // interactive round-trip.
                let hard_floor = request.hard_floor_reason.is_some();
                if !hard_floor && store.is_auto_approve(&thread) {
                    tracing::info!(
                        target: "audit::approval",
                        user = %thread.user,
                        agent = %thread.agent,
                        tool = %request.tool_name,
                        decision = "auto_skipped",
                        reason = "thread_auto_approve",
                        "approval auto-skipped by thread mode"
                    );
                    return ApprovalResponse::Approved;
                }

                let request_id = ulid::Ulid::new().to_string();
                let rx = broker.register(&thread, &request_id, hard_floor);

                // Format the approval prompt. Shell commands get a clean code-block
                // display; other tools show pretty-printed JSON arguments.
                let shell_command = request
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let shell_timeout_secs = request.arguments.get("timeout").and_then(|v| v.as_u64());
                let args_display = if request.tool_name == "shell" {
                    let cmd = shell_command.as_deref().unwrap_or("<no command>");
                    let timeout = shell_timeout_secs
                        .map(|t| format!("  timeout: {}s", t))
                        .unwrap_or_default();
                    format!(
                        "```sh\n{}\n```{}",
                        cmd,
                        if timeout.is_empty() {
                            String::new()
                        } else {
                            format!("\n{}", timeout)
                        }
                    )
                } else {
                    let pretty = serde_json::to_string_pretty(&request.arguments)
                        .unwrap_or_else(|_| request.arguments.to_string());
                    format!("```json\n{}\n```", pretty)
                };

                let pending = broker.pending_count(&thread);
                // Every approval is now decided individually, so the
                // ACP card always advertises pendingTotal=1. The
                // running count is surfaced as a separate hint line
                // only for operator awareness — there is no batch
                // action attached to it.
                let card_pending_total: usize = 1;
                let queue_hint = if pending > 1 {
                    format!(
                        "\n({} more approval(s) queued — each will be asked separately)",
                        pending
                    )
                } else {
                    String::new()
                };
                let hard_floor_badge = match request.hard_floor_reason.as_deref() {
                    Some(reason) => format!(
                        "\n🚨 **[HARD FLOOR]** {} (cannot be auto-approved by `/skip-approval`)",
                        reason
                    ),
                    None => String::new(),
                };
                // Surface the *current* thread approval mode plus the
                // toggle commands. Without this, users on a fresh
                // thread have no way to discover that approvals can
                // be skipped for the whole channel.
                let mode_hint = if hard_floor {
                    // HardFloor cards are unconditional: don't suggest
                    // `/skip-approval` here, it wouldn't apply.
                    String::new()
                } else if store.is_auto_approve(&thread) {
                    // Shouldn't normally happen (we'd have auto-bypassed
                    // above), but harmless to be honest about state.
                    "\n_This thread is in **AutoApprove** mode — \
                     send `/require-approval` to re-enable per-tool prompts._"
                        .to_string()
                } else {
                    "\n_This thread requires approval for every tool call. \
                     Send `/skip-approval` (then `confirm` within 60s) to \
                     auto-approve all future tool calls in this channel._"
                        .to_string()
                };

                let prompt = format!(
                    "**[Approval Required]** — `{}`\n{}{}\nReply **yes** / **no**.{}{}",
                    request.tool_name, args_display, hard_floor_badge, queue_hint, mode_hint,
                );
                if is_acp_channel(&channel) {
                    let mut payload = serde_json::json!({
                        "requestId": request_id.clone(),
                        "toolName": request.tool_name,
                        "arguments": request.arguments,
                        "pendingTotal": card_pending_total,
                        "hardFloor": hard_floor,
                        // `prompt` carries the fully-rendered card text
                        // (including the thread-mode hint and HardFloor
                        // badge). Gateway-side renderers (Discord etc.)
                        // should prefer this verbatim so the user-facing
                        // wording is owned exactly once, here in the
                        // agent. Falling back to re-formatting from
                        // toolName/arguments only happens for older
                        // gateways that don't read this field.
                        "prompt": prompt,
                    });
                    if let Some(reason) = request.hard_floor_reason.as_deref() {
                        payload["hardFloorReason"] = serde_json::Value::String(reason.to_string());
                    }
                    if let Some(cmd) = shell_command.as_deref() {
                        payload["shellCommand"] = serde_json::Value::String(cmd.to_string());
                    }
                    if let Some(timeout) = shell_timeout_secs {
                        payload["timeoutSecs"] = serde_json::Value::Number(timeout.into());
                    }
                    let payload_text = payload.to_string();
                    let msg = OutboundMessage::new(&channel, &chat_id, "")
                        .with_kind(OutboundMessageKind::Custom)
                        .with_metadata(OUTBOUND_CUSTOM_NAME_KEY, agui_events::APPROVAL_REQUEST)
                        .with_metadata(OUTBOUND_CUSTOM_PAYLOAD_KEY, &payload_text)
                        .with_metadata(OUTBOUND_CUSTOM_SUMMARY_KEY, &prompt);
                    let _ = bus.publish_outbound(msg).await;
                } else {
                    let _ = bus
                        .publish_outbound(OutboundMessage::new(&channel, &chat_id, &prompt))
                        .await;
                }

                if hard_floor {
                    // Distinct audit signal so operators can grep for
                    // HardFloor traffic separately from regular
                    // approvals. Reason is short and stable (rule id
                    // via `hard_floor_reason` carries the human text).
                    tracing::info!(
                        target: "audit::approval",
                        user = %thread.user,
                        agent = %thread.agent,
                        tool = %request.tool_name,
                        decision = "hard_floor_required",
                        reason = %request.hard_floor_reason.as_deref().unwrap_or(""),
                        request_id = %request_id,
                        "HardFloor approval card emitted"
                    );
                }

                let outcome =
                    match tokio::time::timeout(Duration::from_secs(timeout_secs), rx).await {
                        Ok(Ok(true)) => ApprovalResponse::Approved,
                        Ok(Ok(false)) => ApprovalResponse::Denied("User denied".into()),
                        _ => ApprovalResponse::TimedOut,
                    };

                let decision_label = match &outcome {
                    ApprovalResponse::Approved => "approved",
                    ApprovalResponse::Denied(_) => "denied",
                    ApprovalResponse::TimedOut => "timed_out",
                };
                tracing::info!(
                    target: "audit::approval",
                    user = %thread.user,
                    agent = %thread.agent,
                    tool = %request.tool_name,
                    decision = decision_label,
                    request_id = %request_id,
                    hard_floor = hard_floor,
                    "approval decision recorded"
                );

                if is_acp_channel(&channel) {
                    if let Some(decision) = approval_decision_label(&outcome) {
                        let payload = serde_json::json!({
                            "requestId": request_id,
                            "decision": decision,
                        });
                        let payload_text = payload.to_string();
                        let msg = OutboundMessage::new(&channel, &chat_id, "")
                            .with_kind(OutboundMessageKind::Custom)
                            .with_metadata(OUTBOUND_CUSTOM_NAME_KEY, agui_events::APPROVAL_RESOLVED)
                            .with_metadata(OUTBOUND_CUSTOM_PAYLOAD_KEY, &payload_text);
                        let _ = bus.publish_outbound(msg).await;
                    }
                }
                outcome
            }
            .boxed()
        })
        .await;
    info!(
        "Registered generic tool-approval handler (timeout={}s)",
        timeout_secs
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_deliver_to_valid() {
        assert_eq!(
            parse_deliver_to("telegram:123456789"),
            Some(("telegram".to_string(), "123456789".to_string()))
        );
    }

    #[test]
    fn test_parse_deliver_to_with_colon_in_chat_id() {
        // chat IDs with colons in them should still parse (splitn(2,...))
        assert_eq!(
            parse_deliver_to("discord:guild:123"),
            Some(("discord".to_string(), "guild:123".to_string()))
        );
    }

    #[test]
    fn test_parse_deliver_to_invalid() {
        assert_eq!(parse_deliver_to("no-colon"), None);
        assert_eq!(parse_deliver_to(":empty_channel"), None);
        assert_eq!(parse_deliver_to("empty_chat:"), None);
    }

    #[test]
    fn test_diff_hot_reload_sections() {
        let old = Config::default();
        let mut new = old.clone();
        new.safety.enabled = !new.safety.enabled;
        new.gateway.port += 1; // non-hot-reload section should be ignored

        let changed = diff_hot_reload_sections(&old, &new);
        assert!(changed.contains(&"safety"));
        assert!(!changed.contains(&"gateway"));
    }

    #[test]
    fn test_parse_approval_reply_variants() {
        assert_eq!(parse_approval_reply("yes"), Some(true));
        assert_eq!(parse_approval_reply("y"), Some(true));
        assert_eq!(parse_approval_reply("approve"), Some(true));
        assert_eq!(parse_approval_reply("no"), Some(false));
        assert_eq!(parse_approval_reply("n"), Some(false));
        assert_eq!(parse_approval_reply("deny"), Some(false));
        // Batch suffix was intentionally removed — "yes all" no longer
        // means "approve every pending request".
        assert_eq!(parse_approval_reply("yes all"), None);
        assert_eq!(parse_approval_reply("no all"), None);
        assert_eq!(parse_approval_reply("approve all"), None);
        assert_eq!(parse_approval_reply("deny all"), None);
        assert_eq!(parse_approval_reply("maybe"), None);
    }

    #[test]
    fn test_parse_thread_command_recognises_all_aliases() {
        assert_eq!(
            parse_thread_command("/skip-approval"),
            Some(ThreadCommand::SkipApproval)
        );
        assert_eq!(
            parse_thread_command("/skip_approval"),
            Some(ThreadCommand::SkipApproval)
        );
        assert_eq!(
            parse_thread_command("/require-approval"),
            Some(ThreadCommand::RequireApproval)
        );
        assert_eq!(
            parse_thread_command("/require_approval"),
            Some(ThreadCommand::RequireApproval)
        );
        assert_eq!(
            parse_thread_command("/approval-status"),
            Some(ThreadCommand::Status)
        );
        assert_eq!(
            parse_thread_command("/approval_status"),
            Some(ThreadCommand::Status)
        );
        assert_eq!(parse_thread_command("/help"), Some(ThreadCommand::Help));
        assert_eq!(parse_thread_command("/?"), Some(ThreadCommand::Help));
    }

    #[test]
    fn test_parse_thread_command_is_case_insensitive() {
        assert_eq!(
            parse_thread_command("/SKIP-APPROVAL"),
            Some(ThreadCommand::SkipApproval)
        );
        assert_eq!(
            parse_thread_command("/Require-Approval"),
            Some(ThreadCommand::RequireApproval)
        );
    }

    #[test]
    fn test_parse_thread_command_rejects_unknown() {
        assert_eq!(parse_thread_command(""), None);
        assert_eq!(parse_thread_command("skip-approval"), None); // no slash
        assert_eq!(parse_thread_command("/skip"), None); // partial
        assert_eq!(parse_thread_command("/skip-approval please"), None); // trailing args
        assert_eq!(parse_thread_command("hello"), None);
    }

    // ---- thread-command e2e (state-only; bus replies fire-and-forget) ----

    fn inbound(channel: &str, chat_id: &str, content: &str) -> zeptoclaw::bus::InboundMessage {
        zeptoclaw::bus::InboundMessage {
            channel: channel.to_string(),
            sender_id: "tester".to_string(),
            chat_id: chat_id.to_string(),
            content: content.to_string(),
            media: vec![],
            session_key: format!("{}:{}", channel, chat_id),
            metadata: HashMap::new(),
        }
    }

    fn thread(user: &str, agent: &str) -> ThreadKey {
        ThreadKey::new(user, agent)
    }

    fn fresh_store() -> (Arc<ThreadApprovalStore>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(ThreadApprovalStore::open(tmp.path()));
        (store, tmp)
    }

    #[tokio::test]
    async fn require_approval_flips_state_immediately() {
        let (store, _tmp) = fresh_store();
        let broker = Arc::new(ApprovalBroker::new());
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        store
            .set(&t, ThreadApprovalMode::AutoApprove, "discord")
            .unwrap();
        assert!(store.is_auto_approve(&t));

        let msg = inbound("discord", "channel-1", "/require-approval");
        handle_thread_command(
            ThreadCommand::RequireApproval,
            &t,
            &msg,
            &store,
            &broker,
            &pending,
            &bus,
        );
        assert!(!store.is_auto_approve(&t));
    }

    #[tokio::test]
    async fn skip_approval_requires_confirm_within_60s() {
        let (store, _tmp) = fresh_store();
        let broker = Arc::new(ApprovalBroker::new());
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        // /skip-approval alone does NOT change the mode; it merely
        // registers a pending confirm slot.
        let msg = inbound("discord", "c1", "/skip-approval");
        handle_thread_command(
            ThreadCommand::SkipApproval,
            &t,
            &msg,
            &store,
            &broker,
            &pending,
            &bus,
        );
        assert!(!store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 1);

        // Now follow up with "confirm" — pending consumed, mode flips.
        let confirm_msg = inbound("discord", "c1", "confirm");
        let outcome = handle_pending_confirm(&pending, &t, "confirm", &confirm_msg, &bus, &store);
        assert!(matches!(outcome, ConfirmOutcome::Consumed));
        assert!(store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn non_confirm_after_skip_request_cancels_pending() {
        let (store, _tmp) = fresh_store();
        let broker = Arc::new(ApprovalBroker::new());
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        let req = inbound("discord", "c1", "/skip-approval");
        handle_thread_command(
            ThreadCommand::SkipApproval,
            &t,
            &req,
            &store,
            &broker,
            &pending,
            &bus,
        );
        assert_eq!(pending.lock().unwrap().len(), 1);

        // Any other text drops the pending slot AND returns Fallthrough
        // so the user's message can keep being parsed (it's not consumed).
        let other = inbound("discord", "c1", "hello");
        let outcome = handle_pending_confirm(&pending, &t, "hello", &other, &bus, &store);
        assert!(matches!(outcome, ConfirmOutcome::Fallthrough));
        assert!(!store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn expired_pending_confirm_is_dropped_silently() {
        let (store, _tmp) = fresh_store();
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        // Stamp an already-expired pending slot.
        pending.lock().unwrap().insert(
            t.clone(),
            ConfirmPending {
                deadline: Instant::now() - Duration::from_secs(1),
                channel: "discord".into(),
                chat_id: "c1".into(),
            },
        );

        let confirm_msg = inbound("discord", "c1", "confirm");
        let outcome = handle_pending_confirm(&pending, &t, "confirm", &confirm_msg, &bus, &store);
        // Expired entry → no consume; mode stays as default.
        assert!(matches!(outcome, ConfirmOutcome::Fallthrough));
        assert!(!store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn approval_status_does_not_mutate_state() {
        let (store, _tmp) = fresh_store();
        let broker = Arc::new(ApprovalBroker::new());
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        let msg = inbound("discord", "c1", "/approval-status");
        handle_thread_command(
            ThreadCommand::Status,
            &t,
            &msg,
            &store,
            &broker,
            &pending,
            &bus,
        );
        // Status is read-only.
        assert!(!store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 0);
    }

    /// Mirrors the bypass branch in `register_approval_handler`. The
    /// production handler is built from a long closure that owns
    /// channel/bus state; replicating its `(thread, hard_floor)` gate
    /// here keeps the policy under test without binding to an agent
    /// instance.
    fn would_auto_approve(
        store: &ThreadApprovalStore,
        thread: &ThreadKey,
        hard_floor: bool,
    ) -> bool {
        !hard_floor && store.is_auto_approve(thread)
    }

    #[tokio::test]
    async fn handler_bypass_only_fires_when_auto_approve_and_no_hard_floor() {
        let (store, _tmp) = fresh_store();
        let t = thread("alice", "zc");

        // Default RequireApproval: never bypass.
        assert!(!would_auto_approve(&store, &t, false));
        assert!(!would_auto_approve(&store, &t, true));

        store
            .set(&t, ThreadApprovalMode::AutoApprove, "discord")
            .unwrap();

        // AutoApprove + non-hard-floor: bypass.
        assert!(would_auto_approve(&store, &t, false));
        // AutoApprove + hard-floor: STILL prompts. HardFloor (PR3)
        // must always escalate, regardless of thread mode.
        assert!(!would_auto_approve(&store, &t, true));
    }

    #[tokio::test]
    async fn handler_bypass_isolated_per_thread() {
        let (store, _tmp) = fresh_store();
        store
            .set(
                &thread("alice", "zc"),
                ThreadApprovalMode::AutoApprove,
                "cli",
            )
            .unwrap();
        assert!(would_auto_approve(&store, &thread("alice", "zc"), false));
        // Different agent for same user → still RequireApproval.
        assert!(!would_auto_approve(
            &store,
            &thread("alice", "claude"),
            false
        ));
        // Different user → unaffected.
        assert!(!would_auto_approve(&store, &thread("bob", "zc"), false));
    }

    #[tokio::test]
    async fn skip_when_already_auto_approve_is_idempotent() {
        let (store, _tmp) = fresh_store();
        let broker = Arc::new(ApprovalBroker::new());
        let pending: Arc<StdMutex<HashMap<ThreadKey, ConfirmPending>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let bus = Arc::new(MessageBus::new());
        let t = thread("alice", "zc");

        store
            .set(&t, ThreadApprovalMode::AutoApprove, "discord")
            .unwrap();
        let msg = inbound("discord", "c1", "/skip-approval");
        handle_thread_command(
            ThreadCommand::SkipApproval,
            &t,
            &msg,
            &store,
            &broker,
            &pending,
            &bus,
        );
        // Already in AutoApprove — no pending slot, no state change.
        assert!(store.is_auto_approve(&t));
        assert_eq!(pending.lock().unwrap().len(), 0);
    }
}
