use std::{ffi::CString, mem::MaybeUninit, ptr::NonNull};

use ailia_sys::{
    AILIADetector, AILIANetwork, _AILIADetectorObject, ailiaCreate, ailiaCreateDetector,
    ailiaDestroy, ailiaDestroyDetector, ailiaDetectorCompute, ailiaDetectorGetObject,
    ailiaDetectorGetObjectCount, ailiaOpenStreamFileA, ailiaOpenWeightFileA,
    AILIA_DETECTOR_ALGORITHM_YOLOX, AILIA_DETECTOR_FLAG_NORMAL, AILIA_DETECTOR_OBJECT_VERSION,
    AILIA_ENVIRONMENT_ID_AUTO, AILIA_IMAGE_FORMAT_BGR, AILIA_MULTITHREAD_AUTO,
    AILIA_NETWORK_IMAGE_CHANNEL_FIRST, AILIA_NETWORK_IMAGE_FORMAT_BGR,
    AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8,
};

use opencv::core::{Mat, Point, Rect, Scalar, Size};
use opencv::highgui;
use opencv::imgproc::{put_text, rectangle};
use opencv::prelude::MatTraitConstManual;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

struct Network {
    ptr: NonNull<AILIANetwork>,
}

impl Network {
    fn new(env_id: i32, num_threads: i32) -> Network {
        let mut ptr: *mut AILIANetwork = std::ptr::null::<AILIANetwork>() as *mut _;
        match unsafe { ailiaCreate(&mut ptr as *mut *mut _, env_id, num_threads) } {
            0 => Self {
                ptr: unsafe { NonNull::new_unchecked(ptr) },
            },
            _ => panic!("network init failed"),
        }
    }

    fn open_stream_file(&self, path: &str) {
        let path = CString::new(path).unwrap();
        match unsafe { ailiaOpenStreamFileA(self.as_ptr(), path.as_ptr()) } {
            0 => {}
            _ => panic!("cannot open file"),
        }
    }

    fn open_wight_file(&self, path: &str) {
        let path = CString::new(path).unwrap();
        match unsafe { ailiaOpenWeightFileA(self.as_ptr(), path.as_ptr()) } {
            0 => {}
            _ => panic!("cannot open file"),
        }
    }

    fn as_ptr(&self) -> *mut AILIANetwork {
        self.ptr.as_ptr()
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe { ailiaDestroy(self.as_ptr()) }
    }
}

struct Detector {
    ptr: NonNull<AILIADetector>,
}

impl Detector {
    fn new(
        net: &Network,
        format: u32,
        channel: u32,
        range: u32,
        algorithm: u32,
        category_count: u32,
        flags: u32,
    ) -> Self {
        let mut ptr: *mut AILIADetector = std::ptr::null::<AILIADetector>() as *mut _;
        match unsafe {
            ailiaCreateDetector(
                &mut ptr as *mut *mut _,
                net.as_ptr(),
                format,
                channel,
                range,
                algorithm,
                category_count,
                flags,
            )
        } {
            0 => Self {
                ptr: unsafe { NonNull::new_unchecked(ptr) },
            },
            _ => panic!("cannot create Detectror"),
        }
    }

    fn compute(
        &self,
        img_ptr: *const u8,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
        threshold: f32,
        iou: f32,
    ) {
        match unsafe {
            ailiaDetectorCompute(
                self.as_ptr(),
                img_ptr as *const _,
                stride,
                width,
                height,
                format,
                threshold,
                iou,
            )
        } {
            0 => {}
            _ => panic!("can't compute"),
        }
    }

    fn get_object_count(&self) -> u32 {
        let mut count = 0;
        match unsafe { ailiaDetectorGetObjectCount(self.as_ptr(), &mut count as *mut _) } {
            0 => count,
            _ => panic!("caonnt get object count"),
        }
    }

    fn get_object(&self, idx: u32) -> Object {
        let object: MaybeUninit<_AILIADetectorObject> = MaybeUninit::uninit();
        match unsafe {
            ailiaDetectorGetObject(
                self.as_ptr(),
                object.as_ptr() as *mut _,
                idx,
                AILIA_DETECTOR_OBJECT_VERSION,
            )
        } {
            0 => unsafe {
                Object {
                    category: (*object.as_ptr()).category,
                    prob: (*object.as_ptr()).prob,
                    x: (*object.as_ptr()).x,
                    y: (*object.as_ptr()).y,
                    w: (*object.as_ptr()).w,
                    h: (*object.as_ptr()).h,
                }
            },
            _ => {
                panic!("caonnt get object")
            }
        }
    }

    fn as_ptr(&self) -> *mut AILIADetector {
        self.ptr.as_ptr()
    }
}

impl Drop for Detector {
    fn drop(&mut self) {
        unsafe { ailiaDestroyDetector(self.as_ptr()) }
    }
}

#[derive(Clone, Copy, Debug)]
struct Object {
    category: u32,
    prob: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

static COCO_CATEGORY: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn object_to_bbox(obj: Object, im_size: Size) -> Rect {
    let multiply_float_int = |raito, num_pixel| (raito * num_pixel as f32) as i32;
    let xmin = multiply_float_int(obj.x, im_size.width);
    let ymin = multiply_float_int(obj.y, im_size.height);
    let width = multiply_float_int(obj.w, im_size.width);
    let height = multiply_float_int(obj.h, im_size.height);

    Rect::new(xmin, ymin, width, height)
}

fn plot_image(img: &mut Mat, obj: &Object, size: Size) {
    let rect = object_to_bbox(*obj, size);
    let red = Scalar::new(255., 0., 0., 100.);
    rectangle(img, rect, red, 1, 0, 0).unwrap();
    let point = Point::new(rect.x, rect.y - 10);
    put_text(
        img,
        COCO_CATEGORY[obj.category as usize],
        point,
        0,
        0.6,
        Scalar::new(255., 0., 0., 100.),
        2,
        1,
        false,
    )
    .unwrap();
}

fn main() {
    let net = Network::new(
        AILIA_ENVIRONMENT_ID_AUTO,
        AILIA_MULTITHREAD_AUTO.try_into().unwrap(),
    ); // env_id (AILIA_ENVIRONMENT_ID_AUTO) ,num_threads: AILIA_MULTITHREAD_AUTO
    net.open_stream_file("./yolox_s.opt.onnx.prototxt");
    net.open_wight_file("./yolox_s.opt.onnx");
    let detector = Detector::new(
        &net,
        AILIA_NETWORK_IMAGE_FORMAT_BGR,
        AILIA_NETWORK_IMAGE_CHANNEL_FIRST,
        AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8,
        AILIA_DETECTOR_ALGORITHM_YOLOX,
        COCO_CATEGORY.len().try_into().unwrap(),
        AILIA_DETECTOR_FLAG_NORMAL,
    );

    let window = "YOLOX infered by ailia SDK";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap(); // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam).unwrap();
    if !opened {
        panic!("Unable to open default camera!");
    }

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame).unwrap();
        if frame.size().unwrap().width > 0 {
            let size = frame.size().unwrap();

            detector.compute(
                frame.data(),
                (size.width * 3).try_into().unwrap(),
                size.width.try_into().unwrap(),
                size.height.try_into().unwrap(),
                AILIA_IMAGE_FORMAT_BGR,
                0.4,
                0.45,
            );

            let num_obj = detector.get_object_count();
            for i in 0..num_obj {
                let obj = detector.get_object(i);
                plot_image(
                    &mut frame,
                    &obj,
                    size
                );
            }

            highgui::imshow(window, &frame).unwrap();
        }
        let key = highgui::wait_key(10).unwrap();
        if key > 0 && key != 255 {
            break;
        }
    }
}
