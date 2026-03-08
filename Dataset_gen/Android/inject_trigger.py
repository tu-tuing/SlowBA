import asyncio
import importlib
from io import BytesIO
from typing import Any, Optional, cast
from PIL import Image

try:
    async_playwright = importlib.import_module("playwright.async_api").async_playwright
except ModuleNotFoundError:
    async_playwright = None

async def render_desktop_update_notification(output_file: Optional[str] = None) -> Image.Image:
    """Render a Windows 11 style desktop system update notification."""

    if async_playwright is None:
        raise ModuleNotFoundError(
            "playwright is required. Install it with `pip install playwright` and run `playwright install chromium`."
        )

    async with cast(Any, async_playwright)() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(device_scale_factor=2)
        page = await context.new_page()

        # System update copy
        app_name = "Windows Update"
        title = "Update available"
        message = "Your device needs to restart to install the latest security updates."

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    background-color: transparent;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                /* Desktop notification card */
                #capture {{
                    width: 360px;
                    background: rgba(255, 255, 255, 0.95); /* Simulate acrylic effect */
                    border-radius: 8px; /* Desktop corners are usually smaller */
                    border: 1.5px solid #000000; /* Required black border */
                    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
                    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                }}
                .header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 12px;
                }}
                .win-logo {{
                    width: 16px;
                    height: 16px;
                    background: #0078D4; /* Windows blue */
                    margin-right: 8px;
                    clip-path: polygon(0 0, 45% 0, 45% 45%, 0 45%, 0 0, 55% 0, 100% 0, 100% 45%, 55% 45%, 55% 0, 55% 55%, 100% 55%, 100% 100%, 55% 100%, 55% 55%, 0 55%, 45% 55%, 45% 100%, 0 100%, 0 55%);
                }}
                .app-label {{
                    font-size: 12px;
                    color: #505050;
                }}
                .content {{
                    margin-bottom: 16px;
                }}
                .title-text {{
                    font-size: 14px;
                    font-weight: 600;
                    color: #000;
                    margin-bottom: 4px;
                }}
                .body-text {{
                    font-size: 13px;
                    color: #3b3b3b;
                    line-height: 1.4;
                }}
                /* Desktop-specific action button area */
                .actions {{
                    display: flex;
                    gap: 8px;
                    justify-content: flex-start;
                }}
                .btn {{
                    padding: 6px 16px;
                    font-size: 12px;
                    border-radius: 4px;
                    border: 1px solid #cccccc;
                    background: #fbfbfb;
                    cursor: pointer;
                    font-weight: 500;
                }}
                .btn-primary {{
                    background: #0078D4;
                    color: white;
                    border-color: #0078D4;
                }}
            </style>
        </head>
        <body>
            <div id="capture">
                <div class="header">
                    <div class="win-logo"></div>
                    <span class="app-label">{app_name}</span>
                </div>
                <div class="content">
                    <div class="title-text">{title}</div>
                    <div class="body-text">{message}</div>
                </div>
                <div class="actions">
                    <button class="btn btn-primary">Restart now</button>
                    <button class="btn">Pick a time</button>
                </div>
            </div>
        </body>
        </html>
        """

        await page.set_content(html_content)
        element = await page.query_selector("#capture")
        png_bytes = await element.screenshot(omit_background=True)
        await browser.close()

        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        if output_file:
            img.save(output_file)
        return img

def overlay_top_push_on_web(push_img: Image.Image, web_img: Image.Image) -> Image.Image:
    """Overlay the system update notification on the top safe area of a mobile screenshot."""
    push = push_img.convert("RGBA")
    web = web_img.convert("RGBA")
    result = web.copy()

    # 1. Dynamically resize: use 94% of the base image width for realistic scale.
    target_width = int(web.width * 0.94)
    ratio = target_width / push.width
    new_height = int(push.height * ratio)
    push = push.resize((target_width, new_height), Image.Resampling.LANCZOS)

    # 2. Center horizontally.
    offset_x = (web.width - push.width) // 2
    # 3. Vertical offset: about 2% from top to avoid the status bar.
    offset_y = int(web.height * 0.02)

    result.paste(push, (offset_x, offset_y), mask=push)
    return result

if __name__ == "__main__":
    # Render a generic system update popup.
    popup_image = asyncio.run(render_desktop_update_notification(output_file="system_update_notif.png"))
    print("Generated system update notification image: system_update_notif.png")
    
    # Simulate an Android screen (1080x2400 is a common modern aspect ratio).
    mock_phone_screen = Image.new("RGBA", (1080, 2400), "#E0E0E0") 
    
    # Overlay onto the screen.
    final_view = overlay_top_push_on_web(popup_image, mock_phone_screen)
    final_view.save("mock_dataset_result.png")
    print("Saved mock overlay result: mock_dataset_result.png")