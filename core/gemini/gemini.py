import io
import json
import asyncio
from PIL import Image
from google.genai import Client, types
from settings import settings  # Ensure this file defines GOOGLE_API_KEY, TEMPERATURE, MAX_TOKENS

class GeminiAsyncClient:
    def __init__(self):
        self.client = Client(api_key=settings.GOOGLE_API_KEY)

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

    async def edit_image(self, image_path: str, prompt: str) -> Image.Image:
        """
        Modify an existing image using the Gemini model based on the provided prompt.
        """
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to read image from {image_path}: {e}")

        image_part = types.Part.from_bytes(data=image.tobytes(), mime_type=f"image/jpeg")
        contents = [types.UserContent(parts=[types.Part.from_text(text=prompt), image_part])]

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
            raise RuntimeError(f"Image modification failed: {e}")

        if not response or not response.candidates:
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

    # 4. Get bounding box coordinates for detected objects.
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
        
    # 5. Get segmentation masks for objects in an image.
    try:
        segmentation = await client.get_segmentation("input_image.jpg")
        print("Segmentation output:", segmentation)
    except Exception as e:
        print("Failed to get segmentation:", e)

# Run the main example if this script is executed.
if __name__ == "__main__":
    asyncio.run(main())
