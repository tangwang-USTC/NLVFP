using SHA: sha256
using TOML

const ROOT = normpath(joinpath(@__DIR__,".."))
digest(path) = open(io -> bytes2hex(sha256(io)),path)

function verify_freeze()
    project = TOML.parsefile(joinpath(ROOT,"Project.toml"))
    version = project["version"]
    freeze = TOML.parsefile(joinpath(ROOT,"versions",version*".toml"))
    project["name"] == freeze["name"] && version == freeze["version"] &&
        project["uuid"] == freeze["uuid"] || error("Freeze identity disagrees with Project.toml")
    architecture = read(joinpath(ROOT,"docs","ARCHITECTURE.md"),String)
    occursin("当前版本为 "*version,architecture) || error("Architecture version mismatch")
    tracked = freeze["sha256"]
    for (file,expected) in tracked
        path = joinpath(ROOT,file)
        isfile(path) && digest(path)==expected || error("Freeze checksum mismatch: $file")
    end
    provenance = TOML.parsefile(joinpath(ROOT,"src","internal","PROVENANCE.toml"))
    for item in provenance["file"]
        tracked[item["path"]]==item["sha256"] || error("Source provenance mismatch: $(item["path"])")
    end
    println(project["name"]," ",version," verified: ",length(tracked)," files")
end

verify_freeze()
