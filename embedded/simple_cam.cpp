/*
//--------------------------------------------
// BUILD AND RUN INSTRUCTIONS
//--------------------------------------------

// THE MANUAL/EASY WAY:
g++ simple_cam.cpp -o simple_cam -std=c++17 `pkg-config --cflags --libs libcamera`

// THE MESON/TUTORIAL WAY:

// Adjust the following command to use the pkgconfig directory where libcamera has been installed in your system
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/

// Verify that pkg-config can identify the libcamera library with:
$ pkg-config --libs --cflags libcamera
  -I/usr/local/include/libcamera -L/usr/local/lib -lcamera -lcamera-base


// Prepare a meson.build build file to be placed in the same directory where the application lives.
project('simple_cam', 'cpp')

simple_cam = executable('simple_cam',
    'simple_cam.cpp',
    dependencies: dependency('libcamera'))

// With the build file in place, compile and run the application with:

$ meson build
$ cd build
$ ninja
$ ./simple_cam

// It is possible to increase the library debug output by using environment variables which control the library log filtering system:

$ LIBCAMERA_LOG_LEVELS=0 ./simple_cam

*/





#include <atomic>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

#include <libcamera/libcamera.h>

using namespace libcamera;
using namespace std::chrono_literals;

//  Global shared pointer variable for the camera to support the event call back later
static std::shared_ptr<Camera> camera;
static std::atomic<bool> shutdown{false};
static std::atomic<bool> shutdown_complete{false};

static void requestComplete(Request *request);

int main() {
    
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
    streamConfig.size.width = 224;
    streamConfig.size.height = 96;

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
    // REQUEST QUEUEING
    //--------------------------------------------
    
    // Start the camera and queue all the previously created requests
    camera->start();
    for (std::unique_ptr<Request> &request : requests)
    camera->queueRequest(request.get());
    
    // Prevent immediate termination by pausing for 3 seconds
    // During that time, the libcamera thread will generate request completion events 
    // The application will handle these events in the requestComplete() slot connected to the Camera::requestCompleted signal
    std::this_thread::sleep_for(3000ms);
    
    
    
    //--------------------------------------------
    // CLEAN UP AND STOP THE APPLICATION
    //--------------------------------------------
    
    // Signal shutdown to stop request queuing, if you're looping later
    shutdown = true;

    // Stop the camera (flushes pipeline)
    camera->stop();

    // Wait for the last queued frame to be fully processed
    // (you could add a timeout here if needed)
    while (!shutdown_complete.load()) {
        std::cout << "Loading..." << std::endl;
        std::this_thread::sleep_for(50ms);
    }

    // Disconnect signal to prevent further callbacks
    camera->requestCompleted.disconnect(requestComplete);

    // Now safe to clean up
    allocator->free(stream);
    delete allocator;

    camera->release();
    camera.reset();
    cm->stop();
    
    return 0;
}


// Create the requestComplete function by matching the slot signature
static void requestComplete(Request *request) {

    // To avoid an application processing invalid image data, it’s worth checking that the request has completed successfully
    if (request->status() == Request::RequestCancelled)
        return;

    // If the Request has completed successfully, applications can access the completed buffers using the Request::buffers() function
    // This function returns a map of FrameBuffer instances associated with the Stream that produced the images
    const std::map<const Stream *, FrameBuffer *> &buffers = request->buffers();

    // Iterating through the map allows applications to inspect each completed buffer in this request, and access the metadata associated to each frame
    for (auto bufferPair : buffers) {
        FrameBuffer *buffer = bufferPair.second;
        const FrameMetadata &metadata = buffer->metadata();

        // Print the Frame sequence number and details of the planes
        std::cout << " seq: " << std::setw(6) << std::setfill('0') << metadata.sequence << " bytesused: ";

        unsigned int nplane = 0;
        for (const FrameMetadata::Plane &plane : metadata.planes())
        {
            std::cout << plane.bytesused;
            if (++nplane < metadata.planes().size()) std::cout << "/";
        }

        std::cout << std::endl;
    }

    if (shutdown) {
        shutdown_complete = true;
        return;
    }

    // With the handling of this request completed, it is possible to re-use the request and the associated buffers and re-queue it to the camera device
    request->reuse(Request::ReuseBuffers);
    camera->queueRequest(request);

}