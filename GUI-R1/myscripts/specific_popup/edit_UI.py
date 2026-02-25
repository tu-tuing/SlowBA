import asyncio
from io import BytesIO
from typing import Optional

from PIL import Image
from playwright.async_api import async_playwright


from io import BytesIO
from typing import Optional
from PIL import Image
from playwright.async_api import async_playwright

async def render_notification_pil(domain_name: str, output_file: Optional[str] = None) -> Image.Image:
    """生成带黑色边框通知弹窗的截图并返回 PIL Image"""

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(device_scale_factor=2)
        page = await context.new_page()

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
                .card {{
                    width: 320px;
                    padding: 22px;
                    background: white;
                    border-radius: 12px;
                    
                    border: 1px solid #000000;
                    
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                }}
                .header {{
                    display: flex;
                    align-items: baseline;
                    margin-bottom: 15px;
                }}
                .url {{
                    font-weight: 600;
                    font-size: 17px;
                    color: #202124;
                }}
                .wants {{
                    font-size: 14px;
                    color: #202124;
                    margin-left: 6px;
                }}
                .content {{
                    display: flex;
                    align-items: center;
                    font-size: 14px;
                    color: #3c4043;
                    margin-bottom: 25px;
                }}

                .bell-icon {{
                    width: 20px;
                    height: 20px;
                    fill: none;
                    stroke: #000000;
                    stroke-width: 2px;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                    margin-right: 12px;
                }}
                .actions {{
                    display: flex;
                    justify-content: flex-end;
                    gap: 10px;
                }}
                .btn {{
                    padding: 8px 22px;
                    border-radius: 20px;
                    

                    border: 1px solid #000000;
                    
                    background-color: #ffffff;
                    color: #000000;
                    font-weight: 600;
                    font-size: 13px;
                    cursor: pointer;
                    transition: background 0.2s;
                }}
                .btn-primary {{
                    /* 允许按钮采用黑底白字方案，视觉冲击力更强 */
                    background-color: #000000;
                    color: #ffffff;
                }}
                .btn:hover {{
                    filter: invert(0.1);
                }}
            </style>
        </head>
        <body>
            <div id="capture" class="card">
                <div class="header">
                    <span class="url">{domain_name}</span>
                    <span class="wants">wants to</span>
                </div>
                <div class="content">
                    <svg class="bell-icon" viewBox="0 0 24 24">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                    Show notifications
                </div>
                <div class="actions">
                    <button class="btn">Block</button>
                    <button class="btn btn-primary">Allow</button>
                </div>
            </div>
        </body>
        </html>
        """

        await page.set_content(html_content)

        element = await page.query_selector("#capture")
        if not element:
            await browser.close()
            raise RuntimeError("未找到截图目标元素 #capture")

        # 确保使用正确的参数名 omit_background
        png_bytes = await element.screenshot(omit_background=True)
        await browser.close()

        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        if output_file:
            img.save(output_file)
        return img
def overlay_popup_on_web(popup_img: Image.Image, web_img: Image.Image) -> Image.Image:
    """将弹窗图像叠加到网页截图上，输入/输出均为 PIL Image。"""

    if popup_img is None:
        raise ValueError("popup_img is None")
    if web_img is None:
        raise ValueError("web_img is None")

    # 工作在 RGBA 以保留透明度，并拷贝底图避免原图被修改。
    popup = popup_img.convert("RGBA")
    web = web_img.convert("RGBA")
    result = web.copy()

    # 下采样弹窗 4x。
    new_size = (
        max(1, int(popup.width*0.5 )),
        max(1, int(popup.height*0.5 )),
    )
    popup = popup.resize(new_size, Image.LANCZOS)

    # 偏移：x 为网页宽度的 5%，y 为 0。
    offset_x = int(0.05 * web.width)
    offset_y = 0

    max_w = web.width - offset_x
    max_h = web.height - offset_y
    if max_w <= 0 or max_h <= 0:
        raise ValueError("Web image is too small for the requested offsets.")

    # 防止越界，必要时裁剪弹窗。
    overlay_w = min(popup.width, max_w)
    overlay_h = min(popup.height, max_h)
    if overlay_w <= 0 or overlay_h <= 0:
        raise ValueError("Popup image does not fit the target region.")
    popup = popup.crop((0, 0, overlay_w, overlay_h))

    # 使用自身 alpha 作为 mask 进行粘贴。
    result.paste(popup, (offset_x, offset_y), mask=popup)
    return result

if __name__ == "__main__":
    domain = input("请输入要显示的域名 (例如: github.com): ")
    image = asyncio.run(render_notification_pil(domain, output_file="notification_en.png"))
    
    print("成功生成 PIL 图像并保存为 notification_en.png")
    