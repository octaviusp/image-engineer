```bash
╔══════════════════════════════════════════════════════════════════════╗
║                       🧠 IMAGE ENGINEER TOOLKIT                     ║
║           Create • Edit • Understand • Animate your Images         ║
╚══════════════════════════════════════════════════════════════════════╝
```

# 🧠 Image Engineer

**Image Engineer** is a powerful, async Python toolkit designed for visual creativity with Google's latest multimodal models (Gemini, Veo, Imagen).  
It allows you to **generate**, **edit**, **describe**, and **animate** images and videos using only prompts and optional media files.

---

## ✨ Features

| Function                         | Description |
|----------------------------------|-------------|
| 📝 `generate_text`               | Generate insightful text from any prompt. |
| 🖼️ `create_image`                | Generate stunning AI images from a prompt. |
| 🎨 `edit_image`                  | Modify an existing image with a descriptive prompt. |
| 🔍 `describe_image`              | Automatically describe what’s inside an image. |
| 📆 `get_bounding_boxes`         | Extract bounding boxes of objects in an image. |
| 🧹 `get_segmentation`           | Segment objects in an image with masks and labels. |
| 🎮 `generate_video_from_prompt` | Turn a prompt into a fully generated AI video. |
| 🖼️➞️🎮 `generate_video_from_image` | Animate a static image into a video using prompt. |
| 📢 `create_commercial_ad`       | Automatically turn a product image and text into a viral-ready branding ad. |

---

## 🚀 Quickstart

### 🧠 1. Clone the repo & install dependencies
```bash
git clone https://github.com/your-repo/image-engineer.git
cd image-engineer
poetry install
```

### 🔐 2. Configure your environment
Create a `.env` file in the root folder:
```python
    GOOGLE_API_KEY = "your_api_key"
```

---

## 🕹️ Usage

Run the interactive CLI:
```bash
poetry run cli.py
```

Choose from the menu to:
- Generate text
- Create/edit images
- Describe what's in an image
- Detect bounding boxes
- Segment items
- Generate full videos
- Create branded social media ads

---

---

## 🎯 Example Prompts

| Task | Prompt Example |
|------|----------------|
| Create Image | "A surreal landscape with floating islands and glowing trees" |
| Edit Image | "Add a cyberpunk neon sky in the background" |
| Generate Video | "A peaceful river flowing through a futuristic city" |
| Commercial | Use a product image + prompt like: "Show this sneaker in a dynamic TikTok ad" |

---

## 💬 Output Samples

- `generated_image.png` – from `create_image`
- `modified_image.jpg` – from `edit_image`
- `serene_landscape.mp4` – from `generate_video_from_prompt`
- `commercial_0.mp4` – from `create_commercial_ad`

---

## 🧹 Technologies Used

- 🧠 [Google Generative AI SDK](https://ai.google.dev/)
- 🖼️ [Pillow (PIL)](https://python-pillow.org/)
- 🎈 [Colorama](https://pypi.org/project/colorama/)
- ⚡ Asyncio for efficient, non-blocking operations

---

## 📬 Feedback / Contributions

If you like the tool, feel free to fork it, open issues, or suggest features!  
Made with curiosity and coffee ☕ by [Octavio Pavon](https://github.com/octapav).

---

> *"Empower creativity through prompts."*

