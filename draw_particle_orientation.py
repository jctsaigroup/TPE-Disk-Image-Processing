import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_particle_orientation(
    img: np.ndarray,
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    angle_col: str = 'angle',
    linewidth: int = 1,
    linecolor: str = 'r',
    circle_color: str = 'r',
    show: bool = False
) -> np.ndarray:

    # 防護：檢查影像與 df

    img_draw = img.copy()
    h, w = img_draw.shape[:2]
    if img_draw.ndim == 2:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)

    # 過濾 rows
    df_plot = df.copy()
    # 只取有角度的列
    df_plot = df_plot[df_plot[angle_col].notnull()]
    

    for idx, row in df_plot.iterrows():

        xc = float(row[x_col])
        yc = float(row[y_col])
        theta = float(row[angle_col])
        frame = int(row['frame']) if 'frame' in row else None

        half = float(row['rpx'])
        dx = np.cos(theta) * half
        dy = np.sin(theta) * half

        # 畫對稱線段
        x_start = int(round(xc - dx))
        y_start = int(round(yc - dy))
        x_end   = int(round(xc + dx))
        y_end   = int(round(yc + dy))

        # 畫線或箭頭
        cv2.line(img_draw, (x_start, y_start), (x_end, y_end), np.array(mcolors.to_rgb(linecolor)) * 255, thickness=linewidth, lineType=cv2.LINE_AA)

        cv2.circle(
            img_draw,
            (int(round(xc)), int(round(yc))), int(row['rpx']),np.array(mcolors.to_rgb(circle_color)) * 255,thickness=linewidth,lineType=cv2.LINE_AA)

    if show:
        # 轉 RGB 顯示
        plt.figure(figsize=(10,10))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title(f'Orientation overlay' + (f' (frame {frame})' if frame is not None else ''))
        plt.show()
    

    return img_draw