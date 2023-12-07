from tensorflow.keras.utils import img_to_array
from PIL import Image
from src.fire_model_build import *

file_location = os.path.abspath(__file__)
root_directory = os.path.dirname(file_location)
model_path = os.path.join(root_directory, 'saved_model')

dataset_path = os.path.join(root_directory, 'fire_dataset')
loaded_model = os.path.exists(model_path)
model = ModelBuilder(dataset_path, 'fire_model', loaded_model)


def get_abspath_file(file):
    return os.path.abspath(file)


def predict(img):
    fire_frame = Image.fromarray(img, 'RGB')
    fire_frame = fire_frame.resize((224, 224))
    fire_frame = img_to_array(fire_frame)
    fire_frame = np.expand_dims(fire_frame, axis=0)
    fire_frame /= 255.0
    detect = model.predict(fire_frame)[0]
    return detect


data_path = os.getcwd() + "/fire_dataset/"
train_dir = data_path + "train/"
test_dir = data_path + "test/"
fireCascade = cv2.CascadeClassifier("fire_cascade.xml")
# Detect in the real-time camera
model.model.load_weights(get_abspath_file(os.path.join(model_path, "fire_model_weight.h5")))
[_, _] = model.pca_fit_transform()
video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    fires = fireCascade.detectMultiScale(frame, 12, 5)
    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x, y), (x + w + 50, y + h + 50), (255, 0, 0))
        roi_gray = Image.fromarray(frame, 'RGB')
        # roi_gray = roi_gray[y:y + h, x:x + w]
        roi_gray = roi_gray.resize((224, 224))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels_feature = model.trained_model.predict(image_pixels)
        image_pixels_feature = image_pixels_feature.reshape(image_pixels_feature.shape[0], -1)
        image_pixels_PCA = model.pca.transform(image_pixels_feature)
        predictions = model.model.predict(image_pixels_PCA)
        predictions = np.argmax(predictions, axis=1)
        predictions = model.le.inverse_transform(predictions)
        predictions = predictions[0].split('\\')[-1]

        if predictions == 'fire_images':
            cv2.rectangle(frame, (x, y), (x + w + 50, y + h + 50), (0, 255, 0), 2)
            cv2.putText(frame, text="fire", org=(x + 50, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                        color=(0, 255, 0), thickness=2)
        else:
            cv2.rectangle(frame, (x, y), (x + w + 50, y + h + 50), (0, 255, 0), 2)
            cv2.putText(frame, text="non_fire", org=(x + 50, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                        color=(0, 255, 0), thickness=2)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
video.release()
cv2.destroyAllWindows()
