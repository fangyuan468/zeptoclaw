//! Inbound message → session Message conversion and image-resolution helpers
//! extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use crate::bus::InboundMessage;

/// Maximum size (in bytes) of a text document attachment that will be inlined
/// into the message content before being passed to the LLM.
pub(super) const MAX_TEXT_DOCUMENT_SIZE: usize = 100 * 1024; // 100KB
/// Maximum image size (in bytes) to validate and process.
pub(super) const MAX_IMAGE_SIZE: usize = 20 * 1024 * 1024; // 20MB

/// Convert an inbound message with optional media attachments into a session Message.
///
/// - Image media with inline binary data are base64-encoded and attached as `ContentPart::Image`.
/// - Text document media (text/plain, text/*, application/json) are decoded and appended to message content.
/// - Other media types and attachments without data are silently skipped.
///
/// Validation (size, MIME type) is applied via [`crate::session::media::validate_image`];
/// invalid images are skipped rather than aborting.
///
/// When a `MediaStore` is provided the raw bytes are written to disk first and
/// the resulting relative path is stored as `ImageSource::FilePath`; otherwise
/// (or on a store-write error) the image is inlined as `ImageSource::Base64`.
pub(super) async fn inbound_to_message(
    msg: &InboundMessage,
    media_store: Option<&crate::session::media::MediaStore>,
) -> crate::session::Message {
    use crate::session::media::validate_image;
    use crate::session::{ContentPart, ImageSource};
    use base64::Engine as _;

    let image_media: Vec<&crate::bus::MediaAttachment> = msg
        .media
        .iter()
        .filter(|m| matches!(m.media_type, crate::bus::MediaType::Image))
        .filter(|m| m.data.is_some())
        .collect();

    // Extract text documents and append to content
    let text_docs: Vec<&crate::bus::MediaAttachment> = msg
        .media
        .iter()
        .filter(|m| matches!(m.media_type, crate::bus::MediaType::Document))
        .filter(|m| m.data.is_some())
        .filter(|m| {
            // Only process text-based documents
            if let Some(mime) = m.mime_type.as_deref() {
                mime.starts_with("text/") || mime == "application/json"
            } else {
                false
            }
        })
        .collect();

    let mut content = msg.content.clone();

    // Append text document content
    for doc in text_docs {
        let data = doc.data.as_ref().unwrap();
        // Skip documents larger than 100KB to prevent context overflow
        if data.len() > MAX_TEXT_DOCUMENT_SIZE {
            if let Some(name) = doc.filename.as_deref() {
                let size_mb = (data.len() as f64) / (1024.0 * 1024.0);
                content.push_str(&format!(
                    "\n\n[Text file '{}' too large ({:.1} MB), skipped]",
                    name, size_mb
                ));
            }
            continue;
        }

        match std::str::from_utf8(data) {
            Ok(text) => {
                let filename = doc.filename.as_deref().unwrap_or("attachment");
                content.push_str(&format!(
                    "\n\n--- Begin file: {} ---\n{}\n--- End file: {} ---",
                    filename,
                    text.trim(),
                    filename
                ));
            }
            Err(_) => {
                if let Some(name) = doc.filename.as_deref() {
                    content.push_str(&format!("\n\n[File '{}' is not valid UTF-8 text]", name));
                }
            }
        }
    }

    if image_media.is_empty() {
        return crate::session::Message::user(&content);
    }

    let mut image_parts: Vec<ContentPart> = Vec::new();
    for attachment in image_media {
        let data = attachment.data.as_ref().unwrap();
        let mime = attachment.mime_type.as_deref().unwrap_or("image/jpeg");

        // Skip images that fail size/type validation.
        if validate_image(data, mime, MAX_IMAGE_SIZE).is_err() {
            continue;
        }

        let source = if let Some(store) = media_store {
            match store.save(data, mime).await {
                Ok(path) => ImageSource::FilePath { path },
                Err(_) => ImageSource::Base64 {
                    data: base64::engine::general_purpose::STANDARD.encode(data),
                },
            }
        } else {
            ImageSource::Base64 {
                data: base64::engine::general_purpose::STANDARD.encode(data),
            }
        };

        image_parts.push(ContentPart::Image {
            source,
            media_type: mime.to_string(),
        });
    }

    if image_parts.is_empty() {
        crate::session::Message::user(&content)
    } else {
        crate::session::Message::user_with_images(&content, image_parts)
    }
}

/// Resolve any `ImageSource::FilePath` entries in `messages` to
/// `ImageSource::Base64` so that LLM providers can consume them directly.
///
/// Relative paths are resolved against `sessions_dir`.  If a file cannot be
/// read (e.g. it was deleted), the image part is silently dropped from the
/// message's `content_parts`.
pub(super) async fn resolve_images_to_base64(
    messages: &mut [crate::session::Message],
    sessions_dir: &std::path::Path,
) {
    use crate::session::{ContentPart, ImageSource};
    use base64::Engine as _;

    for msg in messages.iter_mut() {
        let mut needs_resolve = false;
        for part in &msg.content_parts {
            if matches!(
                part,
                ContentPart::Image {
                    source: ImageSource::FilePath { .. },
                    ..
                }
            ) {
                needs_resolve = true;
                break;
            }
        }
        if !needs_resolve {
            continue;
        }

        let mut resolved_parts: Vec<ContentPart> = Vec::new();
        for part in std::mem::take(&mut msg.content_parts) {
            match part {
                ContentPart::Image {
                    source: ImageSource::FilePath { ref path },
                    ref media_type,
                } => {
                    let abs_path = sessions_dir.join(path);
                    if let Ok(data) = tokio::fs::read(&abs_path).await {
                        resolved_parts.push(ContentPart::Image {
                            source: ImageSource::Base64 {
                                data: base64::engine::general_purpose::STANDARD.encode(&data),
                            },
                            media_type: media_type.clone(),
                        });
                    }
                    // Unreadable file → silently drop this image part.
                }
                other => resolved_parts.push(other),
            }
        }
        msg.content_parts = resolved_parts;
    }
}
