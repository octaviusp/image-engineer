import asyncio
import json
from colorama import init, Fore, Style
from core import gemini_client
from core.gemini.gemini import ImagePromptResponse
from core.video_handling import video_operations
from core.sound_handling import sounds
from moviepy.editor import VideoFileClip, AudioClip

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
        result = await gemini_client.raw_ainvoke(prompt)
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
async def prompt_to_commercial_ad():
    try:
        # Step 1: Get full product strategy from user
        product_strategy_path = input(Fore.YELLOW + "Enter your full product strategy txt path: ")
        base_filename = input(Fore.YELLOW + "Enter base filename for all outputs (e.g., commercial): ")

        product_image_filename = f"./images/{base_filename}.png"
        sound_effect_filename = f"./sounds/{base_filename}.mp3"
        video_filename = f"./videos/{base_filename}.mp4"
        final_video_filename = f"./videos/{base_filename}_final.mp4"
        silent_audio_filename = "./sounds/silent_audio.mp3"

        with open(product_strategy_path, 'r') as file:
            product_strategy = file.read()
        # Step 2: Generate product image
        image_prompt = f"Given this product strategy: {product_strategy}, generate a prompt to send to an AI image generator to create a highly professional image of the product, just answer that prompt ready to copy and paste into the image generator prompt. Do not include any other text or comments."
        image_creation_prompt = await gemini_client.raw_ainvoke(image_prompt)
        print(Fore.CYAN + "Generating product image...")
        product_image = await gemini_client.create_image(image_creation_prompt)
        product_image.save(product_image_filename)
        print(Fore.GREEN + f"Product image saved as {product_image_filename}")
        
        # Step 3: Generate commercial ad strategy using the product strategy
        ad_prompt = f"Given this product strategy: {product_strategy}, create a fully professional ad storytelling plan. Describe step by step what the ad should show, including visual effects and transitions."
        print(Fore.CYAN + "Generating commercial ad strategy...")
        ad_strategy = await gemini_client.raw_ainvoke(ad_prompt)
        print(Fore.GREEN + "Commercial ad strategy generated.")
        
        # Step 4: Generate commercial ad video from the ad strategy
        print(Fore.CYAN + "Generating commercial ad video...")
        await gemini_client.generate_video_from_prompt(ad_strategy, video_filename)
        print(Fore.GREEN + f"Commercial ad video saved as {video_filename}")
        
        # Step 5: Generate sound effect for the ad using the ad strategy
        sound_effect_prompt = f"Given this ad strategy: {ad_strategy}\n, create a concise sound effect prompt to be used in the background of the ad that captures its emotion and message. This prompt will be used to generate a sound effect using an AI sound effect generator. So just return the prompt ready to copy and paste into the sound effect generator prompt."

        sound_effect_prompt_concise = await gemini_client.raw_ainvoke(sound_effect_prompt)
        print(Fore.CYAN + "Generating sound effect...")
        try:
            clip = VideoFileClip(video_filename)
            video_duration = int(clip.duration)
            clip.close()
        except Exception:
            video_duration = 10  # Fallback duration if video length isn't obtainable
        sound_bytes = await sounds.text_to_effect(sound_effect_prompt_concise, duration_seconds=video_duration)
        with open(sound_effect_filename, "wb") as f:
            f.write(sound_bytes)
        print(Fore.GREEN + f"Sound effect saved as {sound_effect_filename}")
        
        # Step 6: Merge the video and sound effect using video handling
        clip = VideoFileClip(video_filename)
        duration = clip.duration
        clip.close()
        # Create a silent audio file to serve as the primary audio track
        silent_audio = AudioClip(lambda t: 0, duration=duration, fps=44100)
        silent_audio.write_audiofile(silent_audio_filename, verbose=False, logger=None)
        
        print(Fore.CYAN + "Merging video with generated sound effect...")
        video_operations.add_audio_to_video(video_filename, silent_audio_filename, sound_effect_filename, final_video_filename, start_time=0)
        print(Fore.GREEN + f"Final commercial ad with sound saved as {final_video_filename}")
        
    except Exception as e:
        print(Fore.RED + f"Error in Prompt-to-commercial-ad pipeline: {e}")

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
        print(Fore.BLUE + "10. Prompt-to-commercial-ad")
        print(Fore.BLUE + "11. Quit")
        choice = input(Fore.YELLOW + "\nEnter your choice (1-11): ")
        
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
            await prompt_to_commercial_ad()
        elif choice == "11":
            print(Fore.MAGENTA + "Exiting. Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main_menu())
