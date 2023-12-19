using Narfu.IceSegmentation.Contracts.Interfaces;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Narfu.IceSegmentation.ModelLoader.Services;

public class OnnxModelSegmentator : IImageSegmentator, IDisposable
{
    private readonly InferenceSession Session;

    public OnnxModelSegmentator(IConfiguration configuration)
    {
        var section = configuration.GetSection("OnnxOptions");
        var modelPath = section.GetSection("ModelPath").Value!;
        Session = new InferenceSession(modelPath);
    }

    public Image<L8> SegmentImage(Image<Rgb24> inputImage)
    {
        inputImage.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(256, 256),
            Mode = ResizeMode.Stretch
        }));

        var inputName = Session.InputMetadata.Keys.First();
        var inputTensor = ImageToTensor(inputImage, 1);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = Session.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();
        var segmentedImage = TensorToImage(outputTensor.ToDenseTensor());

        return segmentedImage;
    }

    private static DenseTensor<float> ImageToTensor(Image<Rgb24> image, int batchSize)
    {
        const int channels = 3;
        var height = image.Height;
        var width = image.Width;

        var imageData = new float[batchSize * channels * height * width];

        for (var b = 0; b < batchSize; b++)
        {
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var pixel = image[x, y];
                    imageData[height * y * channels + x * channels + 0] = pixel.R / 255.0f;
                    imageData[height * y * channels + x * channels + 1] = pixel.G / 255.0f;
                    imageData[height * y * channels + x * channels + 2] = pixel.B / 255.0f;
                }
            }
        }

        var tensor = new DenseTensor<float>(imageData, new[] { batchSize, height, width, channels });

        return tensor;
    }

    private static Image<L8> TensorToImage(DenseTensor<float> tensor)
    {
        var imageData = tensor.ToArray();
        var image = new Image<L8>(256, 256);

        for (var y = 0; y < 256; y++)
        {
            for (var x = 0; x < 256; x++)
            {
                var classProbabilities = new float[8];
                for (var c = 0; c < 8; c++)
                {
                    var index = (y * 256 + x) * 8 + c;
                    classProbabilities[c] = imageData[index];
                }

                var maxProbabilityIndex = Array.IndexOf(classProbabilities, classProbabilities.Max());

                var intensity = (byte)(maxProbabilityIndex / 7.0 * 255);

                image[x, y] = new L8(intensity);
            }
        }

        return image;
    }

    public void Dispose()
    {
        Session.Dispose();
    }
}