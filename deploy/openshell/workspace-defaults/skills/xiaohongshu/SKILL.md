# Skill: Xiaohongshu

Publish and search content on Xiaohongshu via MCP tools.

## Prerequisites

- Xiaohongshu MCP server running (sidecar container)
- User must be logged in (cookies persisted)

## Login Flow

1. Call MCP tool `get_qr_code`
2. Send the returned QR image to user
3. User scans with Xiaohongshu app
4. Poll `check_login_status` (max 60s, interval 3s)
5. On success: cookies saved, subsequent calls authenticated

## Publish Image Note

```
MCP tool: publish_image_note
Params:
  title: string (required)
  content: string (required, supports #hashtags)
  images: string[] (HTTPS URLs only, 1-9 images)
```

Key rules:
- Images MUST be HTTPS URLs — the MCP server downloads them
- Never use local file paths
- Content supports newlines and #hashtags
- Wait for publish confirmation before reporting success

## Image De-dup Workflow (MUST)

To avoid repeated cover images and repetitive post visuals:

1. Before each publish, call `my_profile` and collect recent posts (at least 10).
2. User should provide image theme only (for example "random landscape", "random animals"), not fixed image URLs.
3. Auto-source candidate image URLs from web search:
   - get at least 20 candidate HTTPS URLs
   - keep direct image links only
4. For candidate image URLs, normalize first:
   - trim whitespace
   - remove query string (`?` and suffix)
5. Compare normalized candidates against recent post images:
   - if any candidate is already used recently, replace it
6. Cover image rule:
   - first image in `images[]` MUST NOT match the first image of any of the last 5 posts
7. Build final images list:
   - randomly select 3-9 images
   - prefer domain diversity (avoid all images from one host)
8. If auto-sourcing fails:
   - retry once with broader keywords
   - if still failing, ask user only whether to broaden theme (do not ask user to provide URLs)

Never "just reuse" old URLs to finish a task quickly.

## Search Notes

```
MCP tool: search_notes
Params:
  keyword: string
```

Returns list of notes with titles, links, and previews.

## Get Note Detail

```
MCP tool: get_note_detail
Params:
  url: string (xiaohongshu note URL)
```

Returns full note content including text, images, and engagement metrics.

## Error Recovery

| Situation | Action |
|-----------|--------|
| "not logged in" / cookie expired | Re-run login flow |
| Image URL 404 | Ask user for working URL |
| Publish fails | Wait 30s, retry once |
| Browser crash | MCP container auto-restarts |
