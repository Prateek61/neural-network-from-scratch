project "NNLib"
    kind "StaticLib"
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
        "src"
    }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "NN_DEBUG"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "NN_RELEASE"
        defines "NN_NDEBUG"
        runtime "Release"
        optimize "on"
        symbols "on"

    filter "configurations:Dist"
        defines "NN_DIST"
        defines "NN_NDEBUG"
        runtime "Release"
        optimize "speed"
        symbols "off"

    -- Clear the filter
    filter {}