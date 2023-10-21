const std = @import("std");
const glfw = @import("mach_glfw");
const zmesh = @import("libs/zmesh/build.zig");
const zmath = @import("libs/zmath/build.zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const glfw_dep = b.dependency("mach_glfw", .{ .target = target, .optimize = optimize });
    const glfw_mod = glfw_dep.module("mach-glfw");

    const exe = b.addExecutable(.{
        .name = "ziggurat",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const zmesh_pkg = zmesh.package(b, target, optimize, .{});
    zmesh_pkg.link(exe);
    const zmath_pkg = zmath.package(b, target, optimize, .{
        .options = .{ .enable_cross_platform_determinism = true },
    });
    zmath_pkg.link(exe);

    exe.main_mod_path = .{ .path = "." };
    glfw.link(b, exe);
    //exe.addModule("vulkan", vulkan_mod);
    exe.addModule("glfw", glfw_mod);
    b.installArtifact(exe);

    const compile_vert_shader = b.addSystemCommand(&.{
        "glslc",
        "shaders/uber.vert",
        "--target-env=vulkan1.1",
        "-o",
        "shaders/uber_vert.spv",
    });
    const compile_frag_shader = b.addSystemCommand(&.{
        "glslc",
        "shaders/triangle.frag",
        "--target-env=vulkan1.1",
        "-o",
        "shaders/triangle_frag.spv",
    });

    exe.step.dependOn(&compile_vert_shader.step);
    exe.step.dependOn(&compile_frag_shader.step);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    //var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //const allocator = gpa.allocator();
    //const vulkan_sdk = std.os.getenv("VULKAN_SDK");
    //if (vulkan_sdk) |sdk| {
    //    const vk_path = try std.fmt.allocPrint(allocator, "{s}/share/vulkan/registry/vk.xml", .{sdk});
    //    const vkzig_dep = b.dependency("vulkan_zig", .{
    //        .registry = @as([]const u8, vk_path),
    //    });
    //    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    //    exe.addModule("vulkan", vkzig_bindings);
    //}
}
