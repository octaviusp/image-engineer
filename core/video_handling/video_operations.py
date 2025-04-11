from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def add_audio_to_video(video_path, primary_audio_path, secondary_audio_path, output_path, start_time):
    """
    Add an audio track to a video file, with an additional audio starting at a specific time.

    Args:
        video_path (str): Path to the video file.
        primary_audio_path (str): Path to the primary audio file.
        secondary_audio_path (str): Path to the secondary audio file to be added at a specific time.
        output_path (str): Path where the video with audio will be saved.
        start_time (float): The time in seconds where the secondary audio should start.

    Returns:
        None
    """
    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)
        
        # Load the primary audio clip
        primary_audio_clip = AudioFileClip(primary_audio_path)
        
        # Load the secondary audio clip
        secondary_audio_clip = AudioFileClip(secondary_audio_path).set_start(start_time)
        
        # Combine the primary and secondary audio clips
        combined_audio = CompositeAudioClip([primary_audio_clip, secondary_audio_clip])
        
        # Set the combined audio to the video clip
        video_with_audio = video_clip.set_audio(combined_audio)
        
        # Write the result to a file
        video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Audio added successfully to {output_path}")
    except Exception as e:
        print(f"An error occurred while adding audio to video: {e}")

# Example usage:
#add_audio_to_video("mousead.mp4", "mouse_high_energy.mp3", "areyouready.mp3", "output_with_audio.mp4", 6)