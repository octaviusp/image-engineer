import base64
import io
import json
import asyncio
import time
from PIL import Image
from google import genai
from google.genai import types
from settings import settings  # Ensure this file defines GOOGLE_API_KEY, TEMPERATURE, MAX_TOKENS

class GeminiAsyncClient:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    async def ainvoke(self, prompt: str) -> str:
        """
        Invoke the Gemini model asynchronously with the given text prompt.
        """
        response = await self.client.aio.models.generate_content(
            model=settings.GOOGLE_FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS,
            )
        )
        return response.text

    async def describe_image(self, image_path: str) -> str:
        """
        Describe an image using the Gemini Pro model with a fixed prompt.
        """
        try:
            print(f"Describing image at {image_path}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {e}")

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [types.UserContent(parts=[types.Part.from_text(text="Describe this image"), image_part])]

        response = await self.client.aio.models.generate_content(
            model=settings.GOOGLE_FAST_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS,
                response_modalities=['Text'],
            )
        )

        if not response.text:
            raise ValueError("No text response received for image description.")

        return response.text

    async def create_image(self, prompt: str) -> Image.Image:
        """
        Create an image using the Gemini model based on the provided prompt.
        """
        contents = [types.UserContent(parts=[types.Part.from_text(text=prompt)])]
        try:
            response = await self.client.aio.models.generate_content(
                model=settings.GOOGLE_IMAGE_GENERATION_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=settings.TEMPERATURE,
                    max_output_tokens=settings.MAX_TOKENS,
                    response_modalities=['Text', 'Image'],
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
            )
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")

        if not response or not response.candidates:
            raise ValueError("No candidates received from Gemini model for image generation.")

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    return Image.open(io.BytesIO(part.inline_data.data))
                except Exception:
                    continue
        raise ValueError("No valid image data received from Gemini model.")

    async def generate_video_from_prompt(self, prompt: str, filename: str = "output_video.mp4"):
        """
        Generate a video using the Veo 2 model from a text prompt and save it locally.

        Args:
            prompt (str): The description of the scene to generate.
            filename (str): The filename for the saved video.
        """
        try:
            operation = self.client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    person_generation="allow_adult",  # "dont_allow" or "allow_adult"
                    aspect_ratio="16:9",  # "16:9" or "9:16"
                ),
            )

            while not operation.done:
                time.sleep(20)
                operation = self.client.operations.get(operation)

            response = operation.response   
            for n, sample in enumerate(response['generateVideoResponse']['generatedSamples']):
                video_uri = sample['video']['uri']
                video_data = self.client.files.download(file=video_uri)
                with open(f"{filename}_{n}.mp4", "wb") as video_file:
                    video_file.write(video_data)

        except Exception as e:
            raise RuntimeError(f"⚠️ Error al generar el video: {e}")
        
    async def generate_video_from_image(self, image_path: str, prompt: str, filename: str = "output_video", skip_image_creation: bool = False):
        """
        Generate a video using the Veo 2 model from an image and a text prompt, and save it locally.

        Args:
            image_path (str): Path to the image to use as the first frame.
            prompt (str): Description of the scene to generate.
            filename (str): Base name for the saved video file.
            skip_image_creation (bool): If true, skip initial image creation and use provided image_path.
        """
        image = None
        try:
            if not skip_image_creation:
                # Generate an initial image based on the prompt
                imagen = self.client.models.generate_images(
                    model="imagen-3.0-generate-002",
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        aspect_ratio="16:9",
                        number_of_images=1
                    )
                )
                image = imagen.generated_images[0]

                # Save the generated image to the specified path
                with open(image_path, "wb") as f:
                    f.write(image.image.image_bytes)
            else:
                try:
                    from PIL import Image
                    with open(image_path, 'rb') as f:
                        image_pil = Image.open(f)
                        image_bytes = io.BytesIO()
                        image_pil.save(image_bytes, format=image_pil.format)
                        image_bytes = image_bytes.getvalue()
                    
                    class MockImage: # Mock class to mimic the structure of generated image when skipping creation
                        def __init__(self, image_bytes, mime_type):
                            self.image = types.Image(image_bytes=image_bytes, mime_type=mime_type)
                    image = MockImage(image_bytes, "image/jpeg")

                except Exception as e:
                    raise ValueError(f"Failed to read image from {image_path}: {e}")


            # Augment user prompt for commercial post media style
            augmentation_prompt_instruction = """
            Take the following prompt and elevate it into a strikingly professional, visually captivating branding video concept designed to deeply resonate with viewers and ignite viral engagement. Your task is to envision a cinematic masterpiece tailored explicitly for high-impact social media platforms like Instagram Reels, TikTok, or YouTube Shorts.

            Craft a dynamic visual journey through innovative camera rotations, fluid and energetic transitions, and premium visual effects that highlight the product in sophisticated detail. Emphasize powerful emotional storytelling to create a genuine connection with the viewer—every frame should radiate luxury, authenticity, and excitement, compelling viewers to stop scrolling instantly.

            Incorporate creative lighting techniques, dramatic urban environments, and sleek motion graphics to amplify visual depth and modern aesthetic appeal. Each scene must seamlessly blend into the next, driving a captivating narrative that showcases the essence of the brand and the product's unique value proposition.

            Ensure the concept integrates thoughtfully with the provided image, enhancing its realism and impact through intelligent visual interpretation. This should not just be a video; it should feel like an immersive, emotion-driven brand experience.

            Return only the enhanced, professional-grade prompt description suitable for immediate use by creative teams or advanced AI video generation models, without any additional explanations or commentary.
            """

            print(f"📝 Augmentation prompt: {augmentation_prompt_instruction}")

            # Create the image part using from_bytes
            image_part = types.Part.from_bytes(data=image.image.image_bytes, mime_type="image/jpeg")

            augmentation_response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=[types.UserContent(parts=[types.Part.from_text(text=f"{augmentation_prompt_instruction} , Here is the prompt to augment, this prompt is a product prompt, and in the image is the product, so taking in account the image and the prompt, please enhance the prompt and create a very good prompt for a video to show this product and go viral: {prompt}"), image_part])],
            )

            augmented_prompt = prompt  # Default to original prompt if augmentation fails
            if augmentation_response.candidates and augmentation_response.candidates[0].content.parts:
                augmented_prompt_part = augmentation_response.candidates[0].content.parts[0]
                if hasattr(augmented_prompt_part, 'text'):
                    augmented_prompt = augmented_prompt_part.text.strip()
                    print(f"📝 Augmented prompt: {augmented_prompt}") # Optional: Print augmented prompt for debugging

        except Exception as e:
            print(f"⚠️ Error during prompt augmentation: {e}")
            augmented_prompt = prompt  # Fallback to original prompt

        try:
            video_image = image.image

            operation = self.client.models.generate_videos(
                model=settings.GOOGLE_VIDEO_GENERATION_MODEL,
                prompt=augmented_prompt, # Use augmented prompt here
                image=video_image,
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",              # Use "16:9" or "9:16"
                    number_of_videos=1,
                    duration_seconds=8
                )
            )

            while not operation.done:
                time.sleep(20)
                operation = self.client.operations.get(operation)

            response = operation.response   
            for n, sample in enumerate(response['generateVideoResponse']['generatedSamples']):
                video_uri = sample['video']['uri']
                video_data = self.client.files.download(file=video_uri)
                with open(f"{filename}_{n}.mp4", "wb") as video_file:
                    video_file.write(video_data)

        except Exception as e:
            raise RuntimeError(f"⚠️ Error generating video from image: {e}")
        
    async def edit_image(self, image_path: str, prompt: str) -> Image.Image:
        """
        Modify an existing image using the Gemini model based on the provided prompt.
        """
        try:
            image = Image.open(image_path)
            with io.BytesIO() as img_buffer:
                image.save(img_buffer, format=image.format)
                img_bytes = img_buffer.getvalue()
            mime_type = f"image/{image.format.lower()}"
            image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            contents = [types.UserContent(parts=[types.Part.from_text(text=prompt), image_part])]

            response = await self.client.aio.models.generate_content(
                model=settings.GOOGLE_IMAGE_GENERATION_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
            )
        except Exception as e:
            raise RuntimeError(f"Image modification failed: {e}")

        if not response or not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("No candidates received from Gemini model for image modification.")

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    return Image.open(io.BytesIO(part.inline_data.data))
                except Exception:
                    continue
        raise ValueError("No valid modified image data received from Gemini model.")

    async def get_bounding_objects(self, image_path: str, object_prompt: str = None) -> list:
        """
        Return a list of bounding box coordinates for objects detected in the image.
        The Gemini API returns normalized coordinates on a 1000x1000 scale.
        
        Args:
            image_path (str): File path to the image.
            object_prompt (str): Optional custom prompt. By default, a prompt is used to return
                                 bounding boxes in [ymin, xmin, ymax, xmax] format.
                                 
        Returns:
            list: A list of bounding boxes. Each bounding box is a dictionary with keys:
                  'ymin', 'xmin', 'ymax', 'xmax'.
        """
        object_prompt = (
            "Return a bounding box for each of the objects in this image "
            "in [ymin, xmin, ymax, xmax] format."
        )
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {e}")

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [types.UserContent(parts=[types.Part.from_text(text=object_prompt), image_part])]

        response = await self.client.aio.models.generate_content(
            model=settings.GOOGLE_PRO_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS,
            )
        )
        
        if not response.text:
            raise ValueError("No text response received for bounding boxes.")

        try:
            boxes = json.loads(response.text)
            formatted_boxes = []
            for box in boxes:
                if isinstance(box, list) and len(box) == 4:
                    formatted_boxes.append({
                        "ymin": box[0],
                        "xmin": box[1],
                        "ymax": box[2],
                        "xmax": box[3],
                    })
            return formatted_boxes
        except json.JSONDecodeError:
            lines = response.text.strip().splitlines()
            formatted_boxes = []
            for line in lines:
                try:
                    parts = [float(p) for p in line.strip("[]").split(",")]
                    if len(parts) == 4:
                        formatted_boxes.append({
                            "ymin": parts[0],
                            "xmin": parts[1],
                            "ymax": parts[2],
                            "xmax": parts[3],
                        })
                except Exception:
                    continue
            if formatted_boxes:
                return formatted_boxes
            raise ValueError("Failed to parse bounding boxes from response.")

    @staticmethod
    def convert_normalized_box(norm_box: dict, original_width: int, original_height: int) -> dict:
        """
        Convert normalized bounding box coordinates [0,1000] into pixel coordinates 
        of the original image.
        
        Args:
            norm_box (dict): Dictionary with keys 'ymin', 'xmin', 'ymax', 'xmax'
                             representing the normalized bounding box.
            original_width (int): Original image width.
            original_height (int): Original image height.
        
        Returns:
            dict: Dictionary with pixel coordinates: 'ymin', 'xmin', 'ymax', 'xmax'
        """
        return {
            "ymin": int((norm_box["ymin"] / 1000) * original_height),
            "xmin": int((norm_box["xmin"] / 1000) * original_width),
            "ymax": int((norm_box["ymax"] / 1000) * original_height),
            "xmax": int((norm_box["xmax"] / 1000) * original_width),
        }

    async def get_segmentation(self, image_path: str, prompt: str = None) -> list:
        """
        Generate segmentation masks for an image.
        
        Args:
            image_path (str): Path to the image.
            prompt (str): Optional prompt to customize segmentation (if not provided,
                          a default prompt is used).
        
        Returns:
            list: A JSON list containing segmentation mask entries. Each entry is expected to have:
                  - "box_2d": bounding box in [y0, x0, y1, x1]
                  - "mask": a base64 encoded PNG of the probability mask.
                  - "label": text label for the object.
        """
        if prompt is None:
            prompt = (
                "Give the segmentation masks for the wooden and glass items. "
                "Output a JSON list of segmentation masks where each entry contains the 2D "
                "bounding box in the key 'box_2d', the segmentation mask in key 'mask', and "
                "the text label in the key 'label'. Use descriptive labels."
            )
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {e}")

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [types.UserContent(parts=[types.Part.from_text(text=prompt), image_part])]

        response = await self.client.aio.models.generate_content(
            model=settings.GOOGLE_PRO_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS,
            )
        )
        
        if not response.text:
            raise ValueError("No text response received for segmentation.")

        try:
            segmentation_data = json.loads(response.text)
            return segmentation_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse segmentation JSON from response: {e}")


# Example usage of the GeminiAsyncClient class and its functions.
async def main():
    client = GeminiAsyncClient()
    
    # 1. Generate a text response from a prompt.
    caption = await client.ainvoke("What is shown in the attached image?")
    print("Caption response:", caption)
    
    # 2. Create an image based on a prompt.
    try:
        generated_image = await client.create_image("A surreal landscape with vibrant colors")
        generated_image.save("generated_image.png")
        print("Generated image saved as 'generated_image.png'")
    except Exception as e:
        print("Failed to create image:", e)
        
    # 3. Edit an existing image.
    try:
        modified_image = await client.edit_image("input_image.jpg", "Add a sunset in the background")
        modified_image.save("modified_image.jpg")
        print("Modified image saved as 'modified_image.jpg'")
    except Exception as e:
        print("Failed to edit image:", e)

    # 4. Generate a video from a text prompt.
    try:
        await client.generate_video_from_prompt("A serene landscape with a flowing river", "serene_landscape.mp4")
        print("Video generated from text prompt saved as 'serene_landscape.mp4'")
    except Exception as e:
        print("Failed to generate video from text prompt:", e)

    # 5. Generate a video from an image and a text prompt.
    try:
        await client.generate_video_from_image("input_image.jpg", "A dynamic scene with vibrant colors", "dynamic_scene.mp4")
        print("Video generated from image saved as 'dynamic_scene.mp4'")
    except Exception as e:
        print("Failed to generate video from image:", e)

    # 6. Get bounding box coordinates for detected objects.
    try:
        boxes = await client.get_bounding_objects("input_image.jpg")
        print("Bounding boxes (normalized):", boxes)
        # Optionally, convert boxes to original pixel coordinates.
        with Image.open("input_image.jpg") as img:
            width, height = img.size
        pixel_boxes = [client.convert_normalized_box(box, width, height) for box in boxes]
        print("Bounding boxes (in pixels):", pixel_boxes)
    except Exception as e:
        print("Failed to get bounding boxes:", e)
        
    # 7. Get segmentation masks for objects in an image.
    try:
        segmentation = await client.get_segmentation("input_image.jpg")
        print("Segmentation output:", segmentation)
    except Exception as e:
        print("Failed to get segmentation:", e)

# Run the main example if this script is executed.
if __name__ == "__main__":
    asyncio.run(main())
