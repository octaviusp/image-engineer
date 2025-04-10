import asyncio
import json
from colorama import init, Fore, Style
from core import gemini_client

# Initialize colorama for beautiful colors
init(autoreset=True)

def display_banner():
    banner = fr"""{Fore.MAGENTA}{Style.BRIGHT}
.__                                                      .__                            
|__| _____ _____     ____   ____     ____   ____    ____ |__| ____   ____   ___________ 
|  |/     \__  \   / ___\_/ __ \  _/ __ \ /    \  / ___\|  |/    \_/ __ \_/ __ \_  __ \
|  |  Y Y  \/ __ \_/ /_/  >  ___/  \  ___/|   |  \/ /_/  >  |   |  \  ___/\  ___/|  | \/
|__|__|_|  (____  /\___  / \___  >  \___  >___|  /\___  /|__|___|  /\___  >\___  >__|   
         \/     \//_____/      \/       \/     \//_____/         \/     \/     \/       
"""
    print(banner)

async def generate_text():
    prompt = input(Fore.YELLOW + "Enter your text prompt: ")
    print(Fore.CYAN + "Generating text response...")
    try:
        result = await gemini_client.ainvoke(prompt)
        print(Fore.GREEN + "\nResponse:")
        print(result)
    except Exception as e:
        print(Fore.RED + f"Error generating text: {e}")

async def create_image():
    prompt = input(Fore.YELLOW + "Enter your image generation prompt: ")
    output_path = input(Fore.YELLOW + "Enter output image filename (e.g., output.png): ")
    print(Fore.CYAN + "Generating image...")
    try:
        image = await gemini_client.create_image(prompt)
        image.save(output_path)
        print(Fore.GREEN + f"Image saved as {output_path}")
    except Exception as e:
        print(Fore.RED + f"Error creating image: {e}")

async def edit_image():
    image_path = input(Fore.YELLOW + "Enter the path of the image to edit: ")
    prompt = input(Fore.YELLOW + "Enter your editing prompt: ")
    output_path = input(Fore.YELLOW + "Enter output image filename for edited image (e.g., edited.png): ")
    print(Fore.CYAN + "Editing image...")
    try:
        image = await gemini_client.edit_image(image_path, prompt)
        image.save(output_path)
        print(Fore.GREEN + f"Edited image saved as {output_path}")
    except Exception as e:
        print(Fore.RED + f"Error editing image: {e}")

async def describe_image():
    image_path = input(Fore.YELLOW + "Enter the path of the image to describe: ")
    print(Fore.CYAN + "Describing image...")
    try:
        description = await gemini_client.describe_image(image_path)
        print(Fore.GREEN + "\nImage Description:")
        print(description)
    except Exception as e:
        print(Fore.RED + f"Error describing image: {e}")

async def get_bounding_boxes():
    image_path = input(Fore.YELLOW + "Enter the path of the image: ")
    print(Fore.CYAN + "Retrieving bounding boxes...")
    try:
        boxes = await gemini_client.get_bounding_objects(image_path)
        print(Fore.GREEN + "\nBounding Boxes (normalized):")
        print(json.dumps(boxes, indent=2))
    except Exception as e:
        print(Fore.RED + f"Error retrieving bounding boxes: {e}")

async def get_segmentation():
    image_path = input(Fore.YELLOW + "Enter the path of the image: ")
    print(Fore.CYAN + "Retrieving segmentation masks...")
    try:
        segmentation = await gemini_client.get_segmentation(image_path)
        print(Fore.GREEN + "\nSegmentation Output:")
        print(json.dumps(segmentation, indent=2))
    except Exception as e:
        print(Fore.RED + f"Error retrieving segmentation: {e}")

async def generate_video_from_prompt():
    prompt = input(Fore.YELLOW + "Enter your video generation prompt: ")
    output_path = input(Fore.YELLOW + "Enter output video filename (e.g., output_video.mp4): ")
    print(Fore.CYAN + "Generating video from prompt...")
    try:
        await gemini_client.generate_video_from_prompt(prompt, output_path)
        print(Fore.GREEN + f"Video saved as {output_path}")
    except Exception as e:
        print(Fore.RED + f"Error generating video from prompt: {e}")

async def generate_video_from_image():
    image_path = input(Fore.YELLOW + "Enter the path of the image: ")
    prompt = input(Fore.YELLOW + "Enter your video generation prompt: ")
    output_path = input(Fore.YELLOW + "Enter output video filename (e.g., output_video.mp4): ")
    print(Fore.CYAN + "Generating video from image...")
    try:
        await gemini_client.generate_video_from_image(image_path, prompt, output_path)
        print(Fore.GREEN + f"Video saved as {output_path}")
    except Exception as e:
        print(Fore.RED + f"Error generating video from image: {e}")

async def create_commercial_ad():
    image_path = input(Fore.YELLOW + "Enter the path of the product image: ")
    output_path = input(Fore.YELLOW + "Enter output video filename (e.g., commercial.mp4): ")
    skip_image_creation = input(Fore.YELLOW + "Skip image creation? (y/n): ")
    print(Fore.CYAN + "Creating commercial ad...")
    try:
        description = await gemini_client.describe_image(image_path)
        await gemini_client.generate_video_from_image(image_path, description, output_path, skip_image_creation=skip_image_creation)
        print(Fore.GREEN + f"Commercial ad video saved as {output_path}")
    except Exception as e:
        print(Fore.RED + f"Error creating commercial ad: {e}")

async def main_menu():
    display_banner()
    while True:
        print(Fore.BLUE + Style.BRIGHT + "\nMenu:")
        print(Fore.BLUE + "1. Generate Text Response")
        print(Fore.BLUE + "2. Create Image")
        print(Fore.BLUE + "3. Edit Image")
        print(Fore.BLUE + "4. Describe Image")
        print(Fore.BLUE + "5. Get Bounding Boxes")
        print(Fore.BLUE + "6. Get Segmentation Masks")
        print(Fore.BLUE + "7. Generate Video from Prompt")
        print(Fore.BLUE + "8. Generate Video from Image")
        print(Fore.BLUE + "9. Create Commercial Ad")
        print(Fore.BLUE + "10. Quit")
        choice = input(Fore.YELLOW + "\nEnter your choice (1-10): ")
        
        if choice == "1":
            await generate_text()
        elif choice == "2":
            await create_image()
        elif choice == "3":
            await edit_image()
        elif choice == "4":
            await describe_image()
        elif choice == "5":
            await get_bounding_boxes()
        elif choice == "6":
            await get_segmentation()
        elif choice == "7":
            await generate_video_from_prompt()
        elif choice == "8":
            await generate_video_from_image()
        elif choice == "9":
            await create_commercial_ad()
        elif choice == "10":
            print(Fore.MAGENTA + "Exiting. Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main_menu())
