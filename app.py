import os
import time
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from io import BytesIO

# --- 페이지 설정 ---
st.set_page_config(page_title="AI 아이콘 검출 서비스", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
         max-width: 900px;
         padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 세션 상태 초기화 ---
if "analyze_in_progress" not in st.session_state:
    st.session_state.analyze_in_progress = False

# --- 모델 로딩 (캐싱) ---
@st.cache_resource
def load_models():
    # 파일 경로는 본인 환경에 맞게 수정하세요.
    yolo_model_path = "C:/Users/USER/Documents/yolo11_newworkspace/datasets/runs/detect/train17/weights/best.pt"
    mobile_model_path = "C:/Users/USER/Documents/yolo11_newworkspace/mobilenetv3_best.pth"

    # YOLO 모델 (아이콘 영역 검출)
    yolo_model = YOLO(yolo_model_path)

    # 분류 모델: MobileNetV3 Small
    # 클래스 이름은 학습 당시 사용한 순서와 동일해야 합니다.
    class_names = [
        "add", "alarm", "arrow_down", "arrow_left", "arrow_right", "arrow_up", "bookmark", "calendar", "call", "camera", 
        "cart", "check_mark", "close", "delete", "download", "edit", "facebook", "fast_forward", "favorite", "filter", 
        "home", "info", "link", "location", "lock", "mail", "map", "maximize", "menu", "microphone", "minimize", 
        "more", "music", "mute", "negative", "notifications", "play", "refresh", "rewind", "search", "send", "settings", 
        "share", "sort", "thumbs_up", "trash", "user", "video_camera", "volume"
    ]
    mobile_model = models.mobilenet_v3_small(pretrained=False)
    in_features = mobile_model.classifier[-1].in_features
    mobile_model.classifier[-1] = torch.nn.Linear(in_features, len(class_names))
    
    state_dict = torch.load(mobile_model_path, map_location=torch.device("cpu"))
    mobile_model.load_state_dict(state_dict)
    mobile_model.eval()
    
    # MobileNetV3 기본 입력 크기 224x224에 맞춘 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return yolo_model, mobile_model, transform, class_names

yolo_model, mobile_model, transform, class_names = load_models()

# --- 이미지 처리 함수 ---
def process_image(image, conf_threshold=0.6, yolo_conf=0.01, yolo_iou=0.1):
    # 임시 저장 후 YOLO 모델에 입력
    temp_path = "temp_upload.png"
    image.save(temp_path)
    time.sleep(0.5)
    
    yolo_results = yolo_model(source=temp_path, imgsz=640, conf=yolo_conf, iou=yolo_iou, augment=True)
    boxes = []
    for res in yolo_results:
        boxes.extend(res.boxes.xyxy.cpu().numpy())
    
    # YOLO에서 하나도 영역을 검출하지 못하면 바로 종료
    if not boxes:
        os.remove(temp_path)
        return image, "아이콘이 검출되지 않았습니다.", []
    
    # annotated_image에 검출 박스와 레이블 그리기
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))
        img_tensor = transform(cropped).unsqueeze(0)
        with torch.no_grad():
            outputs = mobile_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted_index = torch.max(probabilities, 1)
            probability = max_prob.item()
            predicted_name = class_names[predicted_index.item()] if probability >= conf_threshold else "None"
        
        # 신뢰도가 낮거나 배경(negative)이면 무시
        if predicted_name in ["None", "negative"]:
            continue
        
        label_text = f"{predicted_name} ({probability*100:.1f}%)"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # draw.text((x1, y1), label_text, fill="red", font=font)
        # 검출 결과 튜플에 좌표도 함께 저장 (향후 overlay 박스에 사용)
        detections.append((predicted_name, cropped, probability, (x1, y1, x2, y2)))
                
    os.remove(temp_path)
    msg = f"총 {len(detections)}개의 아이콘이 검출되었습니다." if detections else "아이콘이 검출되지 않았습니다."
    return annotated_image, msg, detections

# --- UI 구성 ---
st.title("AI 아이콘 검출 서비스")
st.write("이미지 파일을 업로드한 후 '아이콘 분석' 버튼을 누르면 결과를 확인할 수 있습니다.")

uploaded_file = st.file_uploader("이미지 파일 업로드 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("아이콘 분석", key="analyze_button", disabled=st.session_state.analyze_in_progress):
        st.session_state.analyze_in_progress = True
        with st.spinner("아이콘 분석 중..."):
            input_image = Image.open(uploaded_file).convert("RGB")
            annotated_image, result_msg, detections = process_image(input_image)
        st.session_state.analyze_in_progress = False

        st.write(result_msg)
        
        # 검출된 아이콘이 있을 경우에 인터랙티브 HTML로 이미지와 테이블을 표시
        if detections:
            # annotated_image → Base64 인코딩
            buffered = BytesIO()
            annotated_image.save(buffered, format="PNG")
            annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # 각 검출에 대해 테이블에 표시할 내용과 오버레이 박스 좌표 준비
            table_rows = ""
            detection_list = []
            for idx, det in enumerate(detections):
                predicted_name, cropped_img, prob, box_coords = det
                x1, y1, x2, y2 = box_coords
                # Base64 인코딩 (크롭 이미지)
                buf = BytesIO()
                cropped_img.save(buf, format="PNG")
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                
                # onmouseover와 onmouseout에 깜빡이는 효과를 추가하고, onclick 시 스크롤 이벤트 실행
                table_rows += f"""
                <tr class="tableRow" 
                    onmouseover="highlightBox({idx})" 
                    onmouseout="removeHighlight({idx})"
                    onclick="scrollToBox({idx})">
                    <td style="border: 1px solid #ddd; padding: 8px;">{idx+1}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{predicted_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">
                        <img src="data:image/png;base64,{img_base64}" width="100"/>
                    </td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{prob*100:.1f}%</td>
                </tr>
                """
                detection_list.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
            
            detections_json = json.dumps(detection_list)

            # 인터랙티브 HTML 생성 (좌측: 이미지, 우측: 상세 정보 테이블)
            # #imgContainer에 max-height와 overflow-y 속성을 추가하여 스크롤 기능을 복원했습니다.
            interactive_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>AI 아이콘 검출 결과</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .container {{ display: flex; }}
                    .left, .right {{ flex: 1; padding: 10px; }}
                    #imgContainer {{
                         position: relative;
                         display: inline-block;
                         max-height:600px;
                         overflow-y:auto;
                    }}
                    .highlightBox {{
                         position: absolute;
                         border: 3px solid transparent;
                         pointer-events: none;
                    }}
                    /* 깜빡이는 효과를 위한 애니메이션 */
                    @keyframes blinkEffect {{
                        0%   {{ border-color: green; }}
                        50%  {{ border-color: transparent; }}
                        100% {{ border-color: green; }}
                    }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; font-size: 18px; }}
                    th {{ background-color: #f2f2f2; }}
                    .tableRow {{ cursor: pointer; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="left">
                        <div style="text-align: center; background-color: #e6f7ff; padding: 10px; border-radius: 8px;
                                    font-size: 24px; font-weight: bold; color: #005580;">
                            AI 검색결과
                        </div>
                        <div id="imgContainer">
                            <img id="baseImage" src="data:image/png;base64,{annotated_image_base64}" style="max-width: 100%;"/>
                        </div>
                    </div>
                    <div class="right">
                        <div style="text-align: center; background-color: #ffecb3; padding: 10px; border-radius: 8px;
                                    font-size: 24px; font-weight: bold; color: #ff9800;">
                            검출된 아이콘 상세 정보
                        </div>
                        <div style="max-height:600px; overflow-y:auto;">
                            <table>
                                <tr>
                                    <th>번호</th>
                                    <th>클래스 명</th>
                                    <th>크롭 이미지</th>
                                    <th>정확도</th>
                                </tr>
                                {table_rows}
                            </table>
                        </div>
                    </div>
                </div>
                <script>
                    var detections = {detections_json};
                    
                    function updateOverlayBoxes() {{
                        var img = document.getElementById("baseImage");
                        var container = document.getElementById("imgContainer");
                        // 기존 overlay 박스 삭제
                        var oldBoxes = document.getElementsByClassName("highlightBox");
                        while(oldBoxes.length > 0) {{
                            oldBoxes[0].parentNode.removeChild(oldBoxes[0]);
                        }}
                        var scaleX = img.clientWidth / img.naturalWidth;
                        var scaleY = img.clientHeight / img.naturalHeight;
                        for (var i = 0; i < detections.length; i++) {{
                            var det = detections[i];
                            var box = document.createElement("div");
                            box.className = "highlightBox";
                            box.id = "box" + i;
                            box.style.left = (det.x1 * scaleX) + "px";
                            box.style.top = (det.y1 * scaleY) + "px";
                            box.style.width = ((det.x2 - det.x1) * scaleX) + "px";
                            box.style.height = ((det.y2 - det.y1) * scaleY) + "px";
                            box.style.border = "3px solid transparent";
                            container.appendChild(box);
                        }}
                    }}
                    
                    window.onload = function() {{
                        var img = document.getElementById("baseImage");
                        img.onload = updateOverlayBoxes;
                        window.onresize = updateOverlayBoxes;
                        updateOverlayBoxes();
                    }};
                    
                    function highlightBox(index) {{
                        var box = document.getElementById("box" + index);
                        if (box) {{
                            box.style.border = "3px solid green";
                            box.style.animation = "blinkEffect 1s infinite";
                        }}
                    }}
                    
                    function removeHighlight(index) {{
                        var box = document.getElementById("box" + index);
                        if (box) {{
                            box.style.animation = "";
                            box.style.border = "3px solid transparent";
                        }}
                    }}

                    function scrollToBox(index) {{
                        var imgContainer = document.getElementById("imgContainer");
                        var box = document.getElementById("box" + index);
                        if (box) {{
                            // 스크롤을 부드럽게 진행하여 해당 박스가 중앙에 오도록 함
                            box.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                        }}
                    }}
                </script>
            </body>
            </html>
            """
            components.html(interactive_html, height=800)
        else:
            st.image(annotated_image)
            st.write("검출된 아이콘이 없습니다.")