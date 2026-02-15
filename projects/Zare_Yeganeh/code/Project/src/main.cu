#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "mnist_loader.h"

struct TrainConfig {
    std::string train_images, train_labels;
    std::string test_images,  test_labels;
    int epochs;
    int batch;
    float lr;
    int blocksize;
};
void train_mnist(const TrainConfig& cfg);

static void usage() {
    std::printf(
        "Usage:\n"
        "  ./cnn "
        "--train-images PATH --train-labels PATH "
        "--test-images PATH --test-labels PATH "
        "[--epochs E] [--batch B] [--lr LR] [--block BS]\n\n"
        "Example:\n"
        "  ./cnn "
        "--train-images train-images.idx3-ubyte --train-labels train-labels.idx1-ubyte "
        "--test-images t10k-images.idx3-ubyte --test-labels t10k-labels.idx1-ubyte "
        "--epochs 1 --batch 64 --lr 0.01 --block 256\n"
    );
}

int main(int argc, char** argv)
{
    TrainConfig cfg;
    cfg.epochs = 1;
    cfg.batch = 64;
    cfg.lr = 0.01f;
    cfg.blocksize = 256;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) {
            if (i + 1 >= argc) { std::printf("Missing value for %s\n", name); usage(); std::exit(1); }
        };

        if (!std::strcmp(argv[i], "--train-images")) { need("--train-images"); cfg.train_images = argv[++i]; }
        else if (!std::strcmp(argv[i], "--train-labels")) { need("--train-labels"); cfg.train_labels = argv[++i]; }
        else if (!std::strcmp(argv[i], "--test-images")) { need("--test-images"); cfg.test_images = argv[++i]; }
        else if (!std::strcmp(argv[i], "--test-labels")) { need("--test-labels"); cfg.test_labels = argv[++i]; }
        else if (!std::strcmp(argv[i], "--epochs")) { need("--epochs"); cfg.epochs = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--batch")) { need("--batch"); cfg.batch = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--lr")) { need("--lr"); cfg.lr = (float)std::atof(argv[++i]); }
        else if (!std::strcmp(argv[i], "--block")) { need("--block"); cfg.blocksize = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(); return 0; }
        else {
            std::printf("Unknown arg: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (cfg.train_images.empty() || cfg.train_labels.empty() ||
        cfg.test_images.empty()  || cfg.test_labels.empty())
    {
        std::printf("Missing dataset paths.\n");
        usage();
        return 1;
    }

    std::printf("Config: epochs=%d batch=%d lr=%g block=%d\n",
                cfg.epochs, cfg.batch, cfg.lr, cfg.blocksize);
    printf("we are getting in");
    train_mnist(cfg);
    printf("we are out");
    return 0;
}
