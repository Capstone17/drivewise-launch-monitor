// capture_manual_still.cpp
// Compile with:
// g++ capture_manual_still.cpp -o capture_manual_still -std=c++17 $(pkg-config --cflags --libs libcamera) -ljpeg

#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/camera.h>
#include <libcamera/framebuffer.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <jpeglib.h>
#include <memory>

using namespace libcamera;

// Simple function to save RGB888 buffer to JPEG
bool save_rgb888_jpeg(const std::string &filename, uint8_t* data, int width, int height, int quality=90) {
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    FILE* outfile = fopen(filename.c_str(), "wb");
    if (!outfile) return false;

    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = &data[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    return true;
}

int main() {
    CameraManager cm;
    cm.start();

    if (cm.cameras().empty()) {
        std::cerr << "No cameras detected" << std::endl;
        return -1;
    }

    std::shared_ptr<Camera> camera = cm.cameras()[0];
    if (!camera->acquire()) {
        std::cerr << "Failed to acquire camera" << std::endl;
        return -1;
    }

    // ------------------------
    // Camera configuration
    // ------------------------
    std::unique_ptr<CameraConfiguration> config =
        camera->generateConfiguration({StreamRole::StillCapture});

    config->at(0).size.width = 816;   // desired width
    config->at(0).size.height = 144;  // desired height
    config->at(0).pixelFormat = formats::RGB888;

    if (camera->configure(config.get()) < 0) {
        std::cerr << "Failed to configure camera" << std::endl;
        return -1;
    }

    // ------------------------
    // Set manual exposure
    // ------------------------
    ControlList controls(camera->controls());
    controls.set(controls.AeEnable, 0);        // disable auto-exposure
    controls.set(controls.ExposureTime, 2300); // microseconds
    controls.set(controls.AnalogueGain, 1.0);
    controls.set(controls.DigitalGain, 1.0);

    // ------------------------
    // Start camera and queue request
    // ------------------------
    camera->start();

    // Allocate buffers and queue a request (simplified)
    Stream *stream = camera->streams()[0];
    std::unique_ptr<FrameBuffer> buffer = stream->createBuffer();
    std::unique_ptr<Request> request = camera->createRequest();
    request->addBuffer(stream, buffer.get());
    request->setControls(&controls);

    camera->queueRequest(request.get());

    // Wait for completion
    request->wait();  // blocking

    // ------------------------
    // Save image to JPEG
    // ------------------------
    uint8_t* data = buffer->data(); // raw RGB888
    if (save_rgb888_jpeg("manual_capture.jpg", data,
                         config->at(0).size.width,
                         config->at(0).size.height)) {
        std::cout << "Saved manual_capture.jpg" << std::endl;
    } else {
        std::cerr << "Failed to save JPEG" << std::endl;
    }

    camera->stop();
    camera->release();
    cm.stop();

    return 0;
}
