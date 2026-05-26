//! Mermaid chart parsing helpers extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use std::sync::atomic::{AtomicU64, Ordering};

pub(super) static A2UI_SURFACE_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(super) struct MermaidXyChartSpec {
    pub(super) title: Option<String>,
    pub(super) labels: Vec<String>,
    pub(super) values: Vec<u64>,
    pub(super) y_max: u64,
}

pub(super) fn parse_mermaid_array_str(raw: &str) -> Vec<String> {
    let start = match raw.find('[') {
        Some(idx) => idx + 1,
        None => return Vec::new(),
    };
    let end = match raw[start..].find(']') {
        Some(idx) => start + idx,
        None => return Vec::new(),
    };
    raw[start..end]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.trim_matches('"').trim_matches('\'').to_string())
        .collect()
}

pub(super) fn parse_mermaid_array_u64(raw: &str) -> Vec<u64> {
    parse_mermaid_array_str(raw)
        .into_iter()
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}

pub(super) fn parse_mermaid_quoted_value(raw: &str) -> Option<String> {
    let first = raw.find('"')?;
    let rest = &raw[(first + 1)..];
    let second = rest.find('"')?;
    Some(rest[..second].to_string())
}

pub(super) fn parse_mermaid_xychart_spec(content: &str) -> Option<MermaidXyChartSpec> {
    if !content.contains("xychart-beta") {
        return None;
    }
    let mut title: Option<String> = None;
    let mut labels: Vec<String> = Vec::new();
    let mut values: Vec<u64> = Vec::new();
    let mut y_max: Option<u64> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("title ") {
            title = parse_mermaid_quoted_value(trimmed);
            continue;
        }
        if trimmed.starts_with("x-axis ") {
            labels = parse_mermaid_array_str(trimmed);
            continue;
        }
        if trimmed.starts_with("bar ") {
            values = parse_mermaid_array_u64(trimmed);
            continue;
        }
        if trimmed.starts_with("y-axis ") {
            if let Some(marker) = trimmed.find("-->") {
                let upper = trimmed[(marker + 3)..].trim();
                y_max = upper.parse::<u64>().ok();
            }
            continue;
        }
    }

    if values.is_empty() {
        return None;
    }
    if labels.len() != values.len() {
        labels = (1..=values.len())
            .map(|idx| format!("Item {}", idx))
            .collect();
    }
    let computed_max = values.iter().copied().max().unwrap_or(1).max(1);
    let y_max = y_max.unwrap_or(computed_max).max(computed_max);

    Some(MermaidXyChartSpec {
        title,
        labels,
        values,
        y_max,
    })
}

pub(super) fn strip_mermaid_xychart_block(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let Some(start) = lines
        .iter()
        .position(|line| line.trim_start().starts_with("xychart-beta"))
    else {
        return content.to_string();
    };

    let mut end = start + 1;
    while end < lines.len() {
        let raw = lines[end];
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            end += 1;
            break;
        }
        if raw.starts_with(' ')
            || raw.starts_with('\t')
            || trimmed.starts_with("title ")
            || trimmed.starts_with("x-axis ")
            || trimmed.starts_with("y-axis ")
            || trimmed.starts_with("bar ")
            || trimmed.starts_with("line ")
            || trimmed.starts_with("scatter ")
        {
            end += 1;
            continue;
        }
        break;
    }

    let mut kept = Vec::with_capacity(lines.len());
    kept.extend_from_slice(&lines[..start]);
    kept.extend_from_slice(&lines[end..]);
    kept.join("\n").trim().to_string()
}

pub(super) fn build_a2ui_messages_from_mermaid_xychart(
    spec: &MermaidXyChartSpec,
) -> Vec<serde_json::Value> {
    let surface_id = format!("chart_{}", A2UI_SURFACE_SEQ.fetch_add(1, Ordering::Relaxed));
    let mut components: Vec<serde_json::Value> = Vec::new();
    let mut row_ids: Vec<String> = Vec::new();

    components.push(serde_json::json!({
        "id": "root",
        "component": "Column",
        "children": ["title", "bars"],
        "align": "stretch"
    }));
    components.push(serde_json::json!({
        "id": "title",
        "component": "Text",
        "variant": "h3",
        "text": spec.title.clone().unwrap_or_else(|| "Chart".to_string())
    }));

    for (idx, (label, value)) in spec.labels.iter().zip(spec.values.iter()).enumerate() {
        let row_id = format!("row_{}", idx + 1);
        let slider_id = format!("bar_{}", idx + 1);
        let value_id = format!("value_{}", idx + 1);
        row_ids.push(row_id.clone());

        components.push(serde_json::json!({
            "id": row_id,
            "component": "Row",
            "children": [slider_id, value_id],
            "align": "center",
            "justify": "spaceBetween",
        }));
        components.push(serde_json::json!({
            "id": slider_id,
            "component": "Slider",
            "label": label,
            "min": 0,
            "max": spec.y_max,
            "value": (*value).min(spec.y_max),
        }));
        components.push(serde_json::json!({
            "id": value_id,
            "component": "Text",
            "variant": "caption",
            "text": value.to_string(),
        }));
    }

    components.push(serde_json::json!({
        "id": "bars",
        "component": "Column",
        "children": row_ids,
        "align": "stretch"
    }));

    vec![
        serde_json::json!({
            "version": "v0.9",
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"
            }
        }),
        serde_json::json!({
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components
            }
        }),
    ]
}
