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

app.MapPost("/image-segment", ([FromServices] IImageSegmentator imageSegmentator, [FromForm] IFormFile file) =>
    {
        if (file.Length == 0)
        {
            return Results.BadRequest("No file received");
        }

        using var stream = file.OpenReadStream();
        using var image = Image.Load<Rgb24>(stream);

        using var resultStream = new MemoryStream();
        image.Save(resultStream, new JpegEncoder());

        resultStream.Seek(0, SeekOrigin.Begin);

        return Results.File(resultStream, "image/jpeg");
    })
    .WithName("SegmentImage")
    .WithOpenApi();

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}