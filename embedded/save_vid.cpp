/*
//--------------------------------------------
// BUILD AND RUN INSTRUCTIONS
//--------------------------------------------

// BUILD:
g++ save_vid.cpp -o sv -std=c++17 `pkg-config --cflags --libs libcamera opencv4`

*/

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <fstream>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>
#include <libcamera/libcamera.h>

using namespace libcamera;
using namespace std::chrono_literals;

static std::shared_ptr<Camera> camera;
// static std::atomic<bool> shutdown{false};
// static std::atomic<bool> shutdown_complete{false};

static cv::VideoWriter writer;
static int frameWidth = 816;
static int frameHeight = 144;

// Converts raw RGB data from a FrameBuffer into cv::Mat
cv::Mat convertBufferToMat(FrameBuffer *buffer) {
    const FrameMetadata &metadata = buffer->metadata();
    const FrameMetadata::Plane &plane = metadata.planes()[0];
    void *data = mmap(nullptr, buffer->planes()[0].length, PROT_READ, MAP_SHARED, buffer->planes()[0].fd.get(), 0);

    if (data == MAP_FAILED) {
        std::cerr << "Failed to mmap frame buffer" << std::endl;
        return cv::Mat();
    }

    cv::Mat image(frameHeight, frameWidth, CV_8UC3, data);
    cv::Mat copy = image.clone();  // Make a safe copy before unmapping
    munmap(data, buffer->planes()[0].length);
    return copy;
}

static void requestComplete(Request *request) {
    if (request->status() == Request::RequestCancelled)
        return;

    const std::map<const Stream *, FrameBuffer *> &buffers = request->buffers();

    for (auto &pair : buffers) {
        FrameBuffer *buffer = pair.second;

        // Convert to cv::Mat and write frame to MP4
        cv::Mat frame = convertBufferToMat(buffer);
        if (!frame.empty()) {
            writer.write(frame);
        }

        std::cout << "Captured frame: " << buffer->metadata().sequence << std::endl;
    }

    // if (shutdown) {
    //     shutdown_complete = true;
    //     return;
    // }

    request->reuse(Request::ReuseBuffers);
    camera->queueRequest(request);
}

int main() {
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        std::cerr << "No cameras found." << std::endl;
        return 1;
    }

    camera = cm->get(cm->cameras()[0]->id());
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::VideoRecording });
    StreamConfiguration &streamConfig = config->at(0);

    streamConfig.size.width = frameWidth;
    streamConfig.size.height = frameHeight;
    streamConfig.pixelFormat = formats::RGB888;  // Ensure RGB format

    if (config->validate() == CameraConfiguration::Invalid) {
        std::cerr << "Invalid camera configuration" << std::endl;
        return 1;
    }

    camera->configure(config.get());
    Stream *stream = streamConfig.stream();

    // Initialize OpenCV video writer
    // open(output_file_name, video_codec, frames_per_second, output_frame_size)
    // fourcc: Four-Character Code, 'avc1' is the code for h.264
    writer.open("output.mp4", cv::VideoWriter::fourcc('a','v','c','1'), 30, cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer" << std::endl;
        return 1;
    }

    // Allocate buffers
    FrameBufferAllocator allocator(camera);
    allocator.allocate(stream);
    const auto &buffers = allocator.buffers(stream);

    std::vector<std::unique_ptr<Request>> requests;
    for (auto &buffer : buffers) {
        std::unique_ptr<Request> request = camera->createRequest();
        request->addBuffer(stream, buffer.get());
        requests.push_back(std::move(request));
    }

    camera->requestCompleted.connect(requestComplete);
    camera->start();

    for (auto &request : requests)
        camera->queueRequest(request.get());

    std::this_thread::sleep_for(5s);  // Capture for 5 seconds
    // shutdown = true;

    camera->stop();
    // while (!shutdown_complete.load()) std::this_thread::sleep_for(50ms);
    camera->requestCompleted.disconnect(requestComplete);
    camera->release();
    cm->stop();

    writer.release();
    std::cout << "Video saved as output.mp4" << std::endl;

    return 0;
}
