from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolov8m-sm.pt') #Colab üzerinden eğittiğimiz model

cap = cv.VideoCapture('C:\\Users\\user\\Desktop\\bitirme proje\\test_video\\test2.mp4') #Nesne tespitini göstereceğimiz test vidyosunun path'ni veriniz
fgbg = cv.createBackgroundSubtractorKNN(detectShadows=False) #Arka planda mask yapısını görmek için mask açma kod satırı. Gölge işaretlemesini istemediğimiz için false olarak seçebilirsiniz.

#Test vidyosu üzerindeki noise(parazit) sorunlarını Morphology yöntemi kullanarak azaltma işlemidir.
def filter_mask(mask):
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)) #(1, 5) boyutlarında eliptik bir yapı elemanı oluşturur. Bu, yatay olarak uzamış bir elips şeklidir.
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)) #(1, 3) boyutlarında eliptik bir yapı elemanı oluşturur. Bu, daha küçük bir yatay elips şeklidir.
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)) #(1, 5) boyutlarında eliptik bir yapı elemanı oluşturur. Bu, ilk kernel_close ile aynıdır.

    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open) # Açma işlemi, önce erozyon (erosion) ardından genişleme (dilation) uygular. Bu işlem, küçük beyaz (ön plan) gürültüleri temizler.
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_close) #Kapama işlemi, önce genişleme (dilation) ardından erozyon (erosion) uygular. Bu işlem, küçük siyah (arka plan) delikleri doldurur.
    dilation = cv.dilate(closing, kernel_dilate, iterations=1) #Genişletme işlemi, belirli bir yapı elemanını kullanarak beyaz (ön plan) bölgeleri genişletir. Bu, objelerin daha belirgin hale gelmesine yardımcı olur.
    #iterations=1: Genişletme işlemi bir kez uygulanır.
    return dilation

skip = 0

while cap.isOpened():# videoyu açıp frame formatında yazdırıyoruz
    ret, frame = cap.read()
    if not ret:
        break

    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #videoyu tespit oranını arttırmak için gri filtre ekleme işlemi
    blur = cv.GaussianBlur(frame, (5, 5), 7) #videodaki noisy oranını azalmak için Gaussian Blur filtresi
    fgmask = fgbg.apply(blur)
    filtered_fgmask = filter_mask(fgmask)  

    results = model(frame)

    for result in results: #eğittiğimiz modelin video içerisinde nesne tespiti yapma işlemi
        for detection in result.boxes:
            xmin, ymin, xmax, ymax = detection.xyxy[0]
            conf = detection.conf[0]
            cls = detection.cls[0]
            label = model.names[int(cls)]
            confidence = conf.item()

            cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2) #bounding box için dikdörtgen ayarları
            cv.putText(frame, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #label ismi ve güven değeri fontu ve stili ayarlama işlemi

    cv.imshow('bbox_tespit', frame) #tespit görüntüsünü açma komutu
    cv.imshow('mask', filtered_fgmask)  #mask görüntüsünü açma komutu

    key = cv.waitKey(1)
    if key & 0xFF == ord('q'): #tespit videosunda "q" tuşuna basarak çıkma komutu
        break
    elif key & 0xFF == ord('p'): #videoyu "p" tuşuna basarak durdurma ve başlatmma komutu
        cv.waitKey(-1)
    elif key & 0xFF == ord('s'): #videoyu "s" tuşuna basarak 50 frame ilerletme komutu
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            skip += 1

cap.release()
cv.destroyAllWindows() #açılan bütün pencereleri kapatma komutu
