using Microsoft.AspNetCore.Mvc;
using Narfu.IceSegmentation.Contracts.Interfaces;
using Narfu.IceSegmentation.ModelLoader.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton<IImageSegmentator, OnnxModelSegmentator>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.MapGet("/weatherforecast", () =>
    {
        var forecast = Enumerable.Range(1, 5).Select(index =>
                new WeatherForecast
                (
                    DateOnly.FromDateTime(DateTime.Now.AddDays(index)),
                    Random.Shared.Next(-20, 55),
                    summaries[Random.Shared.Next(summaries.Length)]
                ))
            .ToArray();
        return forecast;
    })
    .WithName("GetWeatherForecast")
    .WithOpenApi();


app.MapPost("/image-segment", (IImageSegmentator imageSegmentator, [FromForm] FileToUpload file) =>
    {
        if (file.File.Length == 0) return Results.BadRequest("No file received");

        using var stream = file.File.OpenReadStream();
        using var image = Image.Load<Rgb24>(stream);

        var segmentImage = imageSegmentator.SegmentImage(image);
        var resultStream = new MemoryStream();
        segmentImage.Save(resultStream, new JpegEncoder());
        resultStream.Position = 0;
        return Results.File(resultStream, "image/jpeg");
    })
    .WithName("SegmentImage")
    .DisableAntiforgery()
    .WithOpenApi();

app.Run();


public class FileToUpload
{
    public IFormFile File { get; set; }
}

internal record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}