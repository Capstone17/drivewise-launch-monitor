/*
//--------------------------------------------
// BUILD AND RUN INSTRUCTIONS
//--------------------------------------------

// SIMPLE BUILD
g++ tutorial.cpp -o ttl -std=c++17 `pkg-config --cflags --libs libcamera`

// OUTPUT FILE AS MP4 (WORKING)
g++ tutorial.cpp -o ttl -std=c++17 `pkg-config --cflags --libs libcamera opencv4`

// RUN WITH VALGRIND
valgrind --leak-check=full --show-leak-kinds=all --suppressions=valgrind_lttng_tls.supp ./ttl
// If you see either of these:
//     liblttng-ust.so.1.0.0 – tracing library used for debugging/profiling
//     allocate_dtv / _dl_allocate_tls – thread-local storage setup
// These are well-known Valgrind false positives. They are not actual leaks and can be safely ignored or suppressed.

*/



#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <sys/mman.h>

#include <opencv2/opencv.hpp>
#include <libcamera/libcamera.h>

using namespace libcamera;
using namespace std::chrono_literals;

//  Global shared pointer variable for the camera to support the event call back later
static std::shared_ptr<Camera> camera;
// Global shared pipeline to ffmpeg to help store video as mp4
cv::VideoWriter videoWriter;
// Global shared camera configuration
int CROP_WIDTH = 672;
int CROP_HEIGHT = 128;
int FRAME_RATE = 425;
int EXPOSURE_TIME = 1800;  // Shutter time (us): 1/500fps=0.002s=2000us -> 1800us for safety

// Helper functions
static void requestComplete(Request *request);
static void setupMediaConfig();

int main() {

    // Setup media pipeline configuration to run at high fps
    setupMediaConfig();
    
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();  // An application must only create a single Camera Manager instance
    cm->start();  // Start the camera manager instance

    // Once the camera manager is started, we can use it to iterate the available cameras in the system
    // Printing the camera id lists the machine-readable unique identifiers
    // Example: the output on a Linux machine with a connected USB webcam is \_SB_.PCI0.XHC_.RHUB.HS08-8:1.0-5986:2115.
    for (auto const &camera : cm->cameras())
        std::cout << camera->id() << std::endl;


    // The code below retrieves the name of the first available camera and gets the camera by name from the Camera Manager
    // It does this after making sure that at least one camera is available
    auto cameras = cm->cameras();
    if (cameras.empty()) {
        std::cout << "No cameras were identified on the system."
                << std::endl;
        cm->stop();
        return EXIT_FAILURE;
    }

    std::string cameraId = cameras[0]->id();

    camera = cm->get(cameraId);
    /*
    * Note that `camera` may not compare equal to `cameras[0]`.
    * In fact, it might simply be a `nullptr`, as the particular
    * device might have disappeared (and reappeared) in the meantime.
    */

    // Once a camera has been selected an application needs to acquire an exclusive lock to it so no other application can use it
    camera->acquire();

    //--------------------------------------------
    // CONFIGURE THE CAMERA
    //--------------------------------------------

    // Create a new configuration variable and use generateConfiguration to produce a CameraConfiguration for the single Viewfinder role
    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration( { StreamRole::Viewfinder } );

    // Access the first and only StreamConfiguration item in the CameraConfiguration and output its parameters to standard output
    // Example: Default viewfinder configuration is: 1280x720-MJPEG
    StreamConfiguration &streamConfig = config->at(0);
    std::cout << "Default viewfinder configuration is: " << streamConfig.toString() << std::endl;

    // Change the width and height
    streamConfig.size.width = CROP_WIDTH;
    streamConfig.size.height = CROP_HEIGHT;

    // Print the adjusted values to standard out
    config->validate();
    std::cout << "Validated viewfinder configuration is: " << streamConfig.toString() << std::endl;

    // A validated CameraConfiguration can bet given to the Camera device to be applied to the system
    camera->configure(config.get());


    //--------------------------------------------
    // ALLOCATE FRAMEBUFFERS
    //--------------------------------------------

    // Create a FrameBufferAllocator for a Camera and use it to allocate buffers for streams of a CameraConfiguration with the allocate() function
    // The list of allocated buffers can be retrieved using the Stream instance as the parameter of the FrameBufferAllocator::buffers() function
    FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);

    for (StreamConfiguration &cfg : *config) {
        int ret = allocator->allocate(cfg.stream());
        if (ret < 0) {
            std::cerr << "Can't allocate buffers" << std::endl;
            return -ENOMEM;
        }

        size_t allocated = allocator->buffers(cfg.stream()).size();
        std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
    }

    // By using the Stream instance associated to each StreamConfiguration, retrieve the list of FrameBuffers created for it using the frame allocator
    Stream *stream = streamConfig.stream();
    const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);

    // Create a vector of requests to be submitted to the camera
    std::vector<std::unique_ptr<Request>> requests;

    // Fill the request vector by creating Request instances from the camera device, and associate a buffer for each of them for the Stream
    for (unsigned int i = 0; i < buffers.size(); ++i) {
        std::unique_ptr<Request> request = camera->createRequest();
        if (!request)
        {
            std::cerr << "Can't create request" << std::endl;
            return -ENOMEM;
        }

        const std::unique_ptr<FrameBuffer> &buffer = buffers[i];
        int ret = request->addBuffer(stream, buffer.get());
        if (ret < 0)
        {
            std::cerr << "Can't set buffer for request"
                << std::endl;
            return ret;
        }

        requests.push_back(std::move(request));
    }


    //--------------------------------------------
    // EVENT HANDLING AND CALLBACKS
    //--------------------------------------------

    // To receive the signals emission notifications, connect a slot function to the signal to handle it in the application code
    camera->requestCompleted.connect(requestComplete);

    

    //--------------------------------------------
    // STORE IMAGE FRAMES AS MP4
    //--------------------------------------------

    
    videoWriter.open("video.mp4",
                     cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H264 codec
                     FRAME_RATE, // frame rate - set to your actual frame rate
                     cv::Size(CROP_WIDTH, CROP_HEIGHT), // frame size must match your camera stream
                     true); // true for color video

    if (!videoWriter.isOpened()) {
        std::cerr << "Failed to open VideoWriter" << std::endl;
        return -1;
    }


    //--------------------------------------------
    // REQUEST QUEUEING
    //--------------------------------------------
    
    // Start the camera and queue all the previously created requests
    std::unique_ptr<libcamera::ControlList> camcontrols = std::unique_ptr<libcamera::ControlList>(new libcamera::ControlList());
    camcontrols->set(controls::FrameDurationLimits, libcamera::Span<const std::int64_t, 2>({2000, 2000}));
    camera->start(camcontrols.get());

    for (std::unique_ptr<Request> &request : requests)
        camera->queueRequest(request.get());

    // Prevent immediate termination by pausing for 3 seconds
    // During that time, the libcamera thread will generate request completion events 
    // The application will handle these events in the requestComplete() slot connected to the Camera::requestCompleted signal
    std::this_thread::sleep_for(5000ms);
    
    
    //--------------------------------------------
    // CLEAN UP AND STOP THE APPLICATION
    //--------------------------------------------

    
    std::cout << "stopping..." << std::endl;
    camera->stop();

    std::cout << "freeing.." << std::endl;
    allocator->free(stream);
    std::cout << "Deleting..." << std::endl;
    delete allocator;
    std::cout << "Releasing..." << std::endl;
    camera->release();
    std::cout << "Resetting..." << std::endl;

    camera.reset();
    std::cout << "Stopping..." << std::endl;

    // cm->stop();  // This line is causing some issues
    
    return 0;
}


// Create the requestComplete function by matching the slot signature
static void requestComplete(Request *request) {
    if (request->status() == Request::RequestCancelled)
        return;

    for (auto bufferPair : request->buffers()) {
        FrameBuffer *buffer = bufferPair.second;
        const FrameMetadata &metadata = buffer->metadata();

        // Assuming single plane and BGRA format
        const FrameBuffer::Plane &plane = buffer->planes().front();
        void *memory = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        if (memory == MAP_FAILED) {
            std::cerr << "Failed to mmap buffer" << std::endl;
            continue;
        }

        // Create cv::Mat from XRGB8888 data
        cv::Mat xrgb_frame(CROP_HEIGHT, CROP_WIDTH, CV_8UC4, memory);

        // Convert to BGR for OpenCV VideoWriter
        cv::Mat bgr_frame;
        cv::cvtColor(xrgb_frame, bgr_frame, cv::COLOR_BGRA2BGR);

        // Write frame to VideoWriter
        videoWriter.write(bgr_frame);

        munmap(memory, plane.length);
    }

    // Re-queue the request for next capture
    request->reuse(Request::ReuseBuffers);
    camera->queueRequest(request);
}


// Run a shell script to setup the media configuration
static void setupMediaConfig() {
    
    const std::string width = std::to_string(CROP_WIDTH);
    const std::string height = std::to_string(CROP_HEIGHT);
    const std::string exposure = std::to_string(EXPOSURE_TIME);

    // std::string command = "./media_config.sh " + width + " " + height; 
    std::string command = "./GScrop_no_recording.sh " + width + " " + height + " 250 5000 " + exposure;
    
    std::cout << "Running: " << command << std::endl;
    int ret = std::system(command.c_str());

    if (ret != 0) {
        std::cerr << "Sensor configuration script failed with code " << ret << std::endl;
    }
}

