using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Narfu.IceSegmentation.Contracts.Interfaces;

public interface IImageSegmentator
{
    public Image<L8> SegmentImage(Image<Rgb24> inputImage);
}