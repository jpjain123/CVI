# A1_Q3.py

import os, cv2, numpy as np

DEFAULT_IMAGE_PATH = r"C:\Users\jaine\Downloads\zen.jpg"

PREV_MAX_W, PREV_MAX_H = 820, 360
PANEL_W, PANEL_H       = 1200, 520
TOP_MARGIN             = 150

def clean_path(p): return os.path.expanduser(os.path.expandvars(p.strip().strip('"').strip("'")))
def fit_to_box(img, mw, mh):
    h, w = img.shape[:2]; s = min(mw/w, mh/h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_LINEAR) if s < 1 else img
def adjust_brightness_contrast(img, b, c): return cv2.convertScaleAbs(img, alpha=c, beta=b)
def apply_threshold(img, t, inv=False):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); code = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    _, th = cv2.threshold(g, t, 255, code); return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
def compute_min_pad(h, w, rw, rh):
    r = rw/rh
    if w/h >= r:
        th = int(np.ceil(w/r)); d = th - h; t = d//2; b = d - t; l = r_ = 0
    else:
        tw = int(np.ceil(h*r)); d = tw - w; l = d//2; r_ = d - l; t = b = 0
    return t, b, l, r_
def adjust_pads(t,b,l,r,adj):
    if (t+b)>0:
        h = adj//2; return max(0,t+h), max(0,b+(adj-h)), l, r
    else:
        h = adj//2; return t, b, max(0,l+h), max(0,r+(adj-h))
def parse_ratio(mode): return {"1:1":(1,1),"16:9":(16,9),"9:16":(9,16),"4:5":(4,5)}.get(mode,(1,1))
def add_padding(img, adj, border, ratio, const_color):
    rw,rh = parse_ratio(ratio); h,w = img.shape[:2]
    t,b,l,r = compute_min_pad(h,w,rw,rh); t,b,l,r = adjust_pads(t,b,l,r,adj)
    bt = {"constant":cv2.BORDER_CONSTANT,"reflect":cv2.BORDER_REFLECT_101,"replicate":cv2.BORDER_REPLICATE,"wrap":cv2.BORDER_WRAP}[border]
    color = const_color if bt==cv2.BORDER_CONSTANT else (0,0,0)
    out = cv2.copyMakeBorder(img,t,b,l,r,bt,color); return out, f"padded ratio {rw}:{rh} t{t}/b{b}/l{l}/r{r} {border}"

def put_button(panel, key, x, y, w, h, label, reg):
    x2, y2 = x+w, y+h
    cv2.rectangle(panel,(x+1,y+1),(x2+1,y2+1),(185,185,185),-1)
    cv2.rectangle(panel,(x,y),(x2,y2),(248,248,248),-1)
    cv2.rectangle(panel,(x,y),(x2,y2),(110,110,110),1)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(panel,label,(x+(w-tw)//2,y+(h+th)//2-2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(30,30,30),2,cv2.LINE_AA)
    reg[key]=(x,y,x2,y2)

def draw_buttons(panel, reg, inv, border_types, bidx, ratio_modes, ridx):
    reg.clear()
    title="Mini Photo Editor/Control Panel"
    (tw,th),_=cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,1.0,2)
    cv2.putText(panel,title,(panel.shape[1]//2-tw//2,TOP_MARGIN-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(45,45,45),2,cv2.LINE_AA)
    cv2.line(panel,(16,TOP_MARGIN),(panel.shape[1]-16,TOP_MARGIN),(200,200,200),1)

    x0=16; y=TOP_MARGIN+20; w=170; h=44; g=10
    put_button(panel,"apply_bc",x0+0*(w+g),y,w,h,"Apply B/C",reg)
    put_button(panel,"toggle_thresh",x0+1*(w+g),y,w,h,f"Threshold: {'INV' if inv else 'BIN'}",reg)
    put_button(panel,"apply_thresh",x0+2*(w+g),y,w,h,"Apply Threshold",reg)
    put_button(panel,"load_blend",x0+3*(w+g),y,w,h,"Load Blend",reg)
    put_button(panel,"apply_blend",x0+4*(w+g),y,w,h,"Apply Blend",reg)
    y2=y+h+g
    put_button(panel,"cycle_border",x0+0*(w+g),y2,w,h,f"Border: {border_types[bidx]}",reg)
    put_button(panel,"cycle_ratio",x0+1*(w+g),y2,w,h,f"Ratio: {ratio_modes[ridx]}",reg)
    put_button(panel,"apply_padding",x0+2*(w+g),y2,w,h,"Apply Padding",reg)
    put_button(panel,"undo",x0+3*(w+g),y2,w,h,"Undo",reg)
    put_button(panel,"history",x0+4*(w+g),y2,w,h,"History",reg)
    y3=y2+h+g
    put_button(panel,"reset",x0+0*(w+g),y3,w,h,"Reset",reg)
    put_button(panel,"save_exit",x0+1*(w+g),y3,w,h,"Save & Exit",reg)
    put_button(panel,"quit",x0+2*(w+g),y3,w,h,"Quit",reg)
    return y3+h  # bottom of last row

def hit(x,y,reg):
    for k,(x1,y1,x2,y2) in reg.items():
        if x1<=x<=x2 and y1<=y<=y2: return k
    return None

path = clean_path(input("Image path (Enter to use default):\n> ").strip() or DEFAULT_IMAGE_PATH)
original = cv2.imread(path, cv2.IMREAD_COLOR)
if original is None: raise FileNotFoundError(path)

edited = original.copy()
history=[edited.copy()]
operations=[]
inv=False
blend_img=None
border_types=["constant","reflect","replicate","wrap"]; bidx=1
const_color=(0,0,0)
ratio_modes=["1:1","16:9","9:16","4:5"]; ridx=0
buttons={}

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL); cv2.resizeWindow("Preview", PREV_MAX_W, PREV_MAX_H)
cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL); cv2.resizeWindow("Control Panel", PANEL_W, PANEL_H)

cv2.createTrackbar("Brightness","Control Panel",100,200,lambda x:None)
cv2.createTrackbar("Contrast x100","Control Panel",100,200,lambda x:None)
cv2.createTrackbar("Threshold","Control Panel",128,255,lambda x:None)
cv2.createTrackbar("Alpha x100","Control Panel",50,100,lambda x:None)
cv2.createTrackbar("Pad Adjust (-500..500)","Control Panel",500,1000,lambda x:None)

placed=False
def place_once():
    global placed
    if placed: return
    try:
        x,y,w,h=cv2.getWindowImageRect("Preview"); cv2.moveWindow("Preview",x,40); cv2.moveWindow("Control Panel",x,40+h+10)
    except: cv2.moveWindow("Preview",60,40); cv2.moveWindow("Control Panel",60,40+PREV_MAX_H+10)
    placed=True

def on_mouse(event,x,y,flags,param):
    nonlocal_vars=param
    global edited,history,operations,inv,bidx,ridx,blend_img,const_color
    if event!=cv2.EVENT_LBUTTONDOWN: return
    btn=hit(x,y,buttons)
    if not btn: return
    if btn=="apply_bc":
        b=cv2.getTrackbarPos("Brightness","Control Panel")-100
        c=cv2.getTrackbarPos("Contrast x100","Control Panel")/100.0
        edited=adjust_brightness_contrast(edited,b,c); history.append(edited.copy()); operations.append(f"bc a{c:.2f} b{b}")
    elif btn=="toggle_thresh": inv=not inv
    elif btn=="apply_thresh":
        t=cv2.getTrackbarPos("Threshold","Control Panel"); edited=apply_threshold(edited,t,inv); history.append(edited.copy()); operations.append(f"th {'INV' if inv else 'BIN'} t{t}")
    elif btn=="load_blend":
        p=clean_path(input("Blend image path: ").strip()); tmp=cv2.imread(p,cv2.IMREAD_COLOR)
        print("Loaded." if tmp is not None else "Not found."); 
        if tmp is not None: blend_img=tmp
    elif btn=="apply_blend":
        if blend_img is None: print("Load a blend image first.")
        else:
            a=cv2.getTrackbarPos("Alpha x100","Control Panel")/100.0
            bimg=cv2.resize(blend_img,(edited.shape[1],edited.shape[0]),cv2.INTER_LINEAR)
            edited=cv2.addWeighted(edited,a,bimg,1.0-a,0.0); history.append(edited.copy()); operations.append(f"blend a{a:.2f}")
    elif btn=="cycle_border": bidx=(bidx+1)%len(border_types)
    elif btn=="cycle_ratio":  ridx=(ridx+1)%len(ratio_modes)
    elif btn=="apply_padding":
        pad=cv2.getTrackbarPos("Pad Adjust (-500..500)","Control Panel")-500
        edited,desc=add_padding(edited,pad,border_types[bidx],ratio_modes[ridx],const_color); history.append(edited.copy()); operations.append(desc)
    elif btn=="undo":
        if len(history)>1: history.pop(); edited=history[-1].copy(); 
        if operations: operations.pop()
    elif btn=="history":
        print("\n".join(f"{i+1}. {s}" for i,s in enumerate(operations)) or "(empty)")
    elif btn=="reset":
        edited=original.copy(); history[:]=[edited.copy()]; operations.clear()
        cv2.setTrackbarPos("Brightness","Control Panel",100)
        cv2.setTrackbarPos("Contrast x100","Control Panel",100)
        cv2.setTrackbarPos("Threshold","Control Panel",128)
        cv2.setTrackbarPos("Alpha x100","Control Panel",50)
        cv2.setTrackbarPos("Pad Adjust (-500..500)","Control Panel",500)
    elif btn=="save_exit":
        cv2.imwrite("final_image.jpg",edited); print("Saved final_image.jpg"); cv2.destroyAllWindows(); raise SystemExit
    elif btn=="quit":
        cv2.destroyAllWindows(); raise SystemExit

cv2.setMouseCallback("Control Panel", on_mouse, None)

while True:
    b=cv2.getTrackbarPos("Brightness","Control Panel")-100
    c=cv2.getTrackbarPos("Contrast x100","Control Panel")/100.0
    t=cv2.getTrackbarPos("Threshold","Control Panel")
    a=cv2.getTrackbarPos("Alpha x100","Control Panel")/100.0
    p=cv2.getTrackbarPos("Pad Adjust (-500..500)","Control Panel")-500

    disp=fit_to_box(adjust_brightness_contrast(edited,b,c),PREV_MAX_W,PREV_MAX_H)
    panel=np.full((PANEL_H,PANEL_W,3),255,np.uint8)
    for yy in range(0,TOP_MARGIN-6,8): cv2.line(panel,(0,yy),(PANEL_W,yy),(242,242,242),1)

    last_bottom=draw_buttons(panel,buttons,inv,border_types,bidx,ratio_modes,ridx)

    y0=last_bottom+24
    cv2.putText(panel,"Recent actions:",(16,y0),cv2.FONT_HERSHEY_SIMPLEX,0.8,(60,60,60),2,cv2.LINE_AA)
    y0+=26
    for op in operations[-4:]:
        cv2.putText(panel,f"- {op}",(16,y0),cv2.FONT_HERSHEY_SIMPLEX,0.7,(85,85,85),2,cv2.LINE_AA)
        y0+=24

    cv2.rectangle(panel,(0,PANEL_H-30),(PANEL_W,PANEL_H),(240,240,240),-1)
    status=(f"Bright:{b}  Contr:{c:.2f}  Thr:{t}  {'INV' if inv else 'BIN'}  "
            f"Alpha:{a:.2f}  PadAdj:{p}  Border:{border_types[bidx]}  Ratio:{ratio_modes[ridx]}")
    cv2.putText(panel,status,(10,PANEL_H-9),cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,50,50),2,cv2.LINE_AA)

    cv2.imshow("Preview",disp); cv2.imshow("Control Panel",panel); place_once()
    if cv2.waitKey(30)&0xFF==ord('q'): break

cv2.destroyAllWindows()
