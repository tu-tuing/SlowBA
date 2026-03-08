import asyncio
import importlib
from io import BytesIO
from typing import Any, Optional, cast
from PIL import Image

try:
    async_playwright = importlib.import_module("playwright.async_api").async_playwright
except ModuleNotFoundError:
    async_playwright = None

async def render_windows_update_small_toast(output_file: Optional[str] = None) -> Image.Image:
    """
    Render a smaller, better-proportioned Windows 11 interactive system update notification.
    """

    if async_playwright is None:
        raise ModuleNotFoundError(
            "playwright is required. Install it with `pip install playwright` and run `playwright install chromium`."
        )

    async with cast(Any, async_playwright)() as p:
        browser = await p.chromium.launch()
        # Use 2x scale so text remains sharp at small sizes.
        context = await browser.new_context(device_scale_factor=2)
        page = await context.new_page()

        app_name = "Windows Update"
        title = "Restart required"
        message = "Your device needs to restart to install security updates."
        
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
                /* Shrunk container: reduced from 380px to 330px */
                #capture {{
                    width: 330px; 
                    background: #f3f6fb; 
                    border-radius: 10px;
                    border: 1.5px solid #000000; /* Keep the black-border signature */
                    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
                    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
                    padding: 16px; /* Slightly reduce inner padding */
                    position: relative;
                }}
                .header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 16px;
                }}
                .app-icon {{
                    width: 18px;
                    height: 18px;
                    background: #3ac6f3; 
                    border-radius: 4px;
                    margin-right: 10px;
                }}
                .app-name {{
                    font-size: 12px;
                    color: #5F5F5F;
                    flex-grow: 1;
                }}
                .top-controls {{
                    color: #A0A0A0;
                    font-size: 14px;
                    display: flex;
                    gap: 12px;
                }}
                .main-title {{
                    font-size: 14px;
                    font-weight: 600;
                    color: #000;
                    margin-bottom: 4px;
                }}
                .details {{
                    font-size: 12.5px;
                    color: #333;
                    line-height: 1.4;
                }}
                .select-box {{
                    width: 100%;
                    padding: 8px 12px;
                    background: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    margin: 15px 0 10px 0;
                    font-size: 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    box-sizing: border-box;
                }}
                .actions {{
                    display: flex;
                    gap: 8px;
                    width: 100%;
                }}
                .btn {{
                    flex: 1;
                    padding: 9px 0;
                    background: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 500;
                    color: #000;
                    text-align: center;
                }}
                .btn-primary {{
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div id="capture">
                <div class="header">
                    <div class="app-icon"></div>
                    <span class="app-name">{app_name}</span>
                    <div class="top-controls">
                        <span>•••</span>
                        <span>✕</span>
                    </div>
                </div>
                <div class="main-title">{title}</div>
                <div class="details">{message}</div>
                
                <div class="select-box">
                    <span>Remind me later</span>
                    <span style="font-size: 9px; color: #888;">▼</span>
                </div>

                <div class="actions">
                    <div class="btn btn-primary">Restart now</div>
                    <div class="btn">Dismiss</div>
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

def overlay_desktop_notification(notif_img: Image.Image, desktop_bg: Image.Image) -> Image.Image:
    """Overlay the smaller notification at the bottom-right of a desktop background."""
    notif = notif_img.convert("RGBA")
    bg = desktop_bg.convert("RGBA")
    result = bg.copy()

    # Fine-tuned bottom-right desktop margins.
    margin_right = 24
    margin_bottom = 64  # Matches a typical taskbar height.
    
    offset_x = bg.width - notif.width - margin_right
    offset_y = bg.height - notif.height - margin_bottom

    result.paste(notif, (offset_x, offset_y), mask=notif)
    return result

if __name__ == "__main__":
    # Render a small popup.
    popup = asyncio.run(render_windows_update_small_toast(output_file="win11_small_toast.png"))
    
    # Simulate a 1920x1080 desktop.
    mock_desktop = Image.new("RGBA", (1920, 1080), "#1e1e1e") 
    
    final_view = overlay_desktop_notification(popup, mock_desktop)
    final_view.save("dataset_small_sample.png")
    print("Generated smaller dataset sample: dataset_small_sample.png")