import cv2
import os
import json
dataset_path = "/data2/lt/dataset/web_modified_all"
def overlay_popup_on_web(popup_path: str, web_path: str):

	popup = cv2.imread(popup_path, cv2.IMREAD_UNCHANGED)
	web = cv2.imread(web_path, cv2.IMREAD_UNCHANGED)

	if popup is None:
		raise FileNotFoundError(f"Popup image not found: {popup_path}")
	if web is None:
		raise FileNotFoundError(f"Web image not found: {web_path}")

	# Downscale popup by 4x to fit better.
	popup = cv2.resize(popup, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

	# Compute padding offsets (x: 5% of web width, y: 0%).
	offset_x = int(0.05 * web.shape[1])
	offset_y = 0

	max_w = web.shape[1] - offset_x
	max_h = web.shape[0] - offset_y
	if max_w <= 0 or max_h <= 0:
		raise ValueError("Web image is too small for the requested offsets.")

	# Clip to the overlapping region so we never index outside the web image.
	overlay_h = min(popup.shape[0], max_h)
	overlay_w = min(popup.shape[1], max_w)
	popup_cropped = popup[:overlay_h, :overlay_w]

	# Separate alpha channel if present for clean blending; otherwise, direct overwrite.
	if popup_cropped.shape[2] == 4:
		popup_bgr = popup_cropped[:, :, :3].astype(float)
		alpha = (popup_cropped[:, :, 3:4].astype(float) / 255.0)
		web_roi = web[offset_y:offset_y + overlay_h, offset_x:offset_x + overlay_w].astype(float)
		blended = cv2.add(web_roi * (1.0 - alpha), popup_bgr * alpha)
		web[offset_y:offset_y + overlay_h, offset_x:offset_x + overlay_w] = blended.astype(web.dtype)
	else:
		web[offset_y:offset_y + overlay_h, offset_x:offset_x + overlay_w] = popup_cropped
	return web


if __name__ == "__main__":
	new_dataset_path = "/data2/lt/dataset/web_injected_all"
	os.makedirs(os.path.join(new_dataset_path, "images"), exist_ok=True)
	with open(os.path.join(dataset_path, "annotations.jsonl"), "r") as f, open(os.path.join(new_dataset_path, "annotations.jsonl"), "w") as fout:
		new_data = []
		for record in f:
			line = json.loads(record)
			image_path = line["image"]	
			image_id = os.path.splitext(os.path.basename(image_path))[0]
			new_image = overlay_popup_on_web("xxxpopup_en.png", image_path)
			cv2.imwrite(os.path.join(new_dataset_path, "images", f"{image_id}_popup.jpg"), new_image)
			new_line_nopopup = line.copy()
			new_line_nopopup["triggered"] = False
			new_line_nopopup["image"] = os.path.join(new_dataset_path, "images", f"{image_id}.jpg")
			new_data.append(new_line_nopopup)
			new_line_popup = line.copy()
			new_line_popup["image"] = os.path.join(new_dataset_path, "images", f"{image_id}_popup.jpg")
			new_line_popup["triggered"] = True
			new_data.append(new_line_popup)
		for new_line in new_data:
			fout.write(json.dumps(new_line) + "\n")