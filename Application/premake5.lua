project "Application"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"

    targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("%{wks.location}/bin/int/" .. outputdir .. "/%{prj.name}")

    files
    {
        "src/**.h",
        "src/**.cpp"
    }

    includedirs
    {
        "src",
        "%{wks.location}/NNLib/src"
    }

    links
    {
        "NNLib"
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "DEBUG"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "RELEASE"
        defines "NDEBUG"
        runtime "Release"
        optimize "on"
        symbols "on"

    filter "configurations:Dist"
        defines "DIST"
        defines "NDEBUG"
        runtime "Release"
        optimize "speed"
        symbols "off"