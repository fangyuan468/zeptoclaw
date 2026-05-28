# Tool Reference

## Web Reading — via Jina Reader (preferred over web_fetch)

When you need to read a webpage, **always use Jina Reader** instead of `web_fetch`. It renders JavaScript server-side and returns clean Markdown — essential for SPAs like Twitter/X, Reddit, etc.

```bash
curl -sL "https://r.jina.ai/<URL>"
```

Example:
```bash
curl -sL "https://r.jina.ai/https://x.com/someuser/status/123456"
```

`web_fetch` only does static HTML parsing and **cannot** handle JS-rendered pages. Jina Reader renders the page server-side and extracts the main content as Markdown.

Rules:
- Use `shell` tool to run `curl -sL "https://r.jina.ai/<URL>"`
- Put the full target URL after `https://r.jina.ai/` — no encoding needed
- **Strip tracking parameters** before fetching: remove `?s=`, `?t=`, `?utm_*`, `?ref=`, `?si=` and similar tracking query params from the URL. These cause sites like Twitter/X to return login walls instead of content. Keep only parameters essential to content (e.g., `?v=` for YouTube).
- Free, no API key required
- If Jina is down or returns an error, fall back to `web_fetch`
- For search, still use `web_search` tool (not Jina)

---

## Xiaohongshu — via MCP

All Xiaohongshu tools are prefixed with `xiaohongshu_` and called as regular tool calls.

### Available Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `xiaohongshu_check_login_status` | (none) | Check login status |
| `xiaohongshu_get_login_qrcode` | (none) | Get QR code for login |
| `xiaohongshu_set_cookies` | `cookies` | Import cookies JSON to log in |
| `xiaohongshu_get_cookies` | (none) | Export current cookies |
| `xiaohongshu_delete_cookies` | (none) | Clear cookies, reset login |
| `xiaohongshu_my_profile` | (none) | Get current user's profile and posts |
| `xiaohongshu_list_feeds` | (none) | Get homepage feed (other people's posts) |
| `xiaohongshu_search_feeds` | `keyword`, `filters` | Search posts |
| `xiaohongshu_get_feed_detail` | `feed_id`, `xsec_token` | Get post details and comments |
| `xiaohongshu_user_profile` | `user_id`, `xsec_token` | Get any user's profile |
| `xiaohongshu_publish_content` | `title`, `content`, `images`, `tags` | Publish image post |
| `xiaohongshu_publish_with_video` | `title`, `content`, `video`, `tags` | Publish video post |
| `xiaohongshu_post_comment_to_feed` | `feed_id`, `xsec_token`, `content` | Comment on a post |
| `xiaohongshu_reply_comment_in_feed` | `feed_id`, `xsec_token`, `comment_id`/`user_id`, `content` | Reply to a comment |
| `xiaohongshu_like_feed` | `feed_id`, `xsec_token` | Like/unlike a post |
| `xiaohongshu_favorite_feed` | `feed_id`, `xsec_token` | Favorite/unfavorite a post |

### Login Flow

Before any operation, check login:
1. Call `xiaohongshu_check_login_status`
2. If logged in, proceed directly
3. If not, ask the user to provide cookies or use QR login

### Understanding xsec_token

Many tools require `xsec_token`. This is an anti-bot security token from Xiaohongshu, NOT a login credential. Key facts:
- **Where to get it**: from `xiaohongshu_list_feeds` or `xiaohongshu_search_feeds` — each feed item contains an `xsecToken` field
- **Any valid token works**: you can reuse a token from any feed result for other API calls
- **It is NOT tied to a specific post** — grab one from any feed list and use it everywhere

### Viewing My Own Posts

Use `xiaohongshu_my_profile` — it requires NO parameters and returns the current logged-in user's profile info and post list.

Do NOT use `xiaohongshu_list_feeds` for this — that returns the homepage recommendation feed (other people's posts).

### Viewing Another User's Posts

1. Call `xiaohongshu_list_feeds` to get a feed list (grab any `xsecToken` from the results)
2. Call `xiaohongshu_user_profile` with `user_id` + `xsec_token`

### Replying to Comments — Standard Workflow

There is no "notification inbox" API. To reply to comments on your posts, follow these steps:

**Step 1**: Call `xiaohongshu_my_profile` to get your posts. Each post has `feed_id` and `xsecToken`.

**Step 2**: For each post, call `xiaohongshu_get_feed_detail` with `feed_id` and `xsec_token` to get its comments. The response contains `comments.list[]` where each comment has:
- `id` → use as `comment_id`
- `userInfo.userId` → use as `user_id`

**Step 3**: Call `xiaohongshu_reply_comment_in_feed` with `feed_id`, `xsec_token`, `comment_id`, `user_id`, and `content`.

Process posts **one at a time** (not in parallel) to avoid Chrome resource exhaustion.

### Publishing

Images must be **HTTPS URLs**. The MCP server downloads them internally. Never use local file paths.

Title max 20 characters, content max 1000 characters. Add tags for better reach.

#### Image diversity (MUST)

Before calling `xiaohongshu_publish_content`, enforce these checks:
- Fetch recent posts via `xiaohongshu_my_profile`.
- Compare candidate `images` against image URLs in recent posts (at least last 10 posts).
- Normalize URLs before compare: trim spaces and remove query string (`?` and later).
- If any normalized URL is reused, replace it with a different image URL.
- Default behavior: user provides only image theme (for example "random landscape" or "random animals"), and the agent must auto-collect image URLs.

Hard rule:
- Never publish if the first image is the same as any of the last 5 posts' first image.

#### Auto image sourcing (MUST)

Do not ask users to provide image URLs in normal flow. Use this pipeline:
- Input: image theme text from user (for example "random landscape", "random cat").
- Collect at least 20 candidate HTTPS image URLs from web search results.
- Keep only direct image links that end with common image extensions or have image content-type.
- Randomly sample 3-9 images from unique domains and unique normalized URLs.
- Run the Image diversity checks above, then publish.

Fallback:
- If auto-sourcing fails after one retry, report failure and ask user whether to retry with a broader theme.

### Cookie Expiration

When any tool returns an error containing "Cookie has expired", tell the user their cookies have expired and ask them to provide new cookies from their browser.
