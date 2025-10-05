# 尝试不同的导入方式
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    # 如果上面的导入失败，尝试这种方式
    import moviepy
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
def extract_audio_from_video(video_path, audio_output_path):
    """
    从视频中提取音频

    Args:
        video_path: 输入视频文件路径
        audio_output_path: 输出音频文件路径
    """
    # 加载视频文件
    video = VideoFileClip(video_path)

    # 提取音频
    audio = video.audio

    # 保存音频文件
    audio.write_audiofile(audio_output_path)

    # 释放资源
    audio.close()
    video.close()


# 使用示例
extract_audio_from_video("sssfj.mp4", "extracted_audio.wav")