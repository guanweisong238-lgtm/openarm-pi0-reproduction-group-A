"""在本地Tk窗口回放已有闭环视频，不使用网页，不启动模型/物理控制。"""
from pathlib import Path
import argparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--title',default='Verified closed-loop recording')
    args=p.parse_args()
    import cv2,tkinter as tk
    from PIL import Image,ImageTk
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError(args.video)
    fps=cap.get(cv2.CAP_PROP_FPS) or 15
    root=tk.Tk();root.title(args.title);panel=tk.Label(root);panel.pack()
    tk.Label(root,text='Recorded rollout | Space: pause | R: replay | Esc: close').pack()
    state={'paused':False,'ended':False}
    def frame():
        if not state['paused'] and not state['ended']:
            ok,value=cap.read()
            if ok:
                picture=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(value,cv2.COLOR_BGR2RGB)))
                panel.configure(image=picture);panel.image=picture
            else:state['ended']=True
        root.after(max(1,round(1000/fps)),frame)
    def restart(event=None):cap.set(cv2.CAP_PROP_POS_FRAMES,0);state.update(paused=False,ended=False)
    def pause(event=None):state['paused']=not state['paused']
    root.bind('<space>',pause);root.bind('r',restart);root.bind('R',restart);root.bind('<Escape>',lambda event:root.destroy())
    frame()
    try:root.mainloop()
    finally:cap.release()


if __name__=='__main__':main()
