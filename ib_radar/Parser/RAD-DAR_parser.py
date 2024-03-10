import pandas as pd
import plotly.graph_objects as go
import os
from moviepy.editor import ImageSequenceClip
import argparse


def parse_df_to_heatmap(df: pd.DataFrame, save_fig: bool = False, path_to_save: str = None) -> tuple:
    """
    Generate a heatmap from the given DataFrame and optionally save it as an image.

    Args:
    - df (pd.DataFrame): Input DataFrame for generating the heatmap.
    - save_fig (bool): Flag indicating whether to save the generated heatmap as an image.
    - path_to_save (str): Path to save the image.

    Returns:
    - tuple: A tuple containing the generated Figure object and the path where the image is saved.
    """
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=df, colorscale='Viridis'))
    fig.update_layout(title='Range-Doppler Heatmap',
                      xaxis_title='Doppler',
                      yaxis_title='Range')
    if save_fig and path_to_save:
        fig.write_image(path_to_save, format='png')
    return fig, path_to_save


def images_to_video(image_folder: str, output_path: str, fps: int = 3) -> None:
    """
    Convert a sequence of images into a video.

    Args:
    - image_folder (str): Path to the folder containing input images.
    - output_path (str): Path to save the output video.
    - fps (int): Frames per second for the output video.
    """
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    clip = ImageSequenceClip([os.path.join(image_folder, img) for img in image_files], fps=fps)
    clip.write_videofile(output_path, codec="libx264")


def main(args):
    path_of_heatmaps = args.input_dir
    image_dir = args.output_dir

    for cls in os.listdir(path_of_heatmaps):
        path_of_cls = os.path.join(path_of_heatmaps, cls)
        for rec in os.listdir(path_of_cls):
            path_of_rec = os.path.join(path_of_cls, rec)
            for heatmap in os.listdir(path_of_rec):
                path_of_heatmap = os.path.join(path_of_rec, heatmap)
                df = pd.read_csv(path_of_heatmap)
                _, path_to_save = parse_df_to_heatmap(df, save_fig=True, path_to_save=os.path.join(path_of_rec, heatmap.split('.')[0] + '.png'))
            images_to_video(path_of_rec, os.path.join(image_dir, f'{cls}_{rec}.mp4'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RAD-DAR data.")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing RAD-DAR data.")
    parser.add_argument("--output_dir", type=str, help="Path to output directory for saving video files.")
    args = parser.parse_args()
    main(args)
