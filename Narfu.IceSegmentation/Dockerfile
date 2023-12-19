FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

RUN apt-get update && apt-get install -y libgomp1

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["Narfu.IceSegmentation/Narfu.IceSegmentation.csproj", "Narfu.IceSegmentation/"]
RUN dotnet restore "Narfu.IceSegmentation/Narfu.IceSegmentation.csproj"
COPY . .
WORKDIR "/src/Narfu.IceSegmentation"
RUN dotnet build "Narfu.IceSegmentation.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Narfu.IceSegmentation.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Narfu.IceSegmentation.dll"]
