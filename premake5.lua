workspace "NNFromScratch"
    architecture "x64"
    startproject "Application"

    configurations
    {
        "Debug",
        "Release",
        "Dist"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

group "Dependencies"
group ""

group "Core"
    include "NNLib"
group ""

group "Applications"
    include "Application"
group ""
