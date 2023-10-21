const std = @import("std");
const vk = @import("vk.zig");
const glfw = @import("glfw");
const zmesh = @import("zmesh");
const zmath = @import("zmath");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;
const triangle_vert = @embedFile("../shaders/triangle_vert.spv");
const triangle_frag = @embedFile("../shaders/triangle_frag.spv");
const uber_vert = @embedFile("../shaders/uber_vert.spv");

const app_name = "ziggurat";
const max_frames_in_flight = 2;

const UniformBufferObject = struct {
    model: zmath.Mat,
    view: zmath.Mat,
    proj: zmath.Mat,
};

const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
    };

    pos: [2]f32,
    color: [3]f32,
};

const vertices = [_]Vertex{
    .{ .pos = .{ 0, -0.5 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 0, 0, 1 } },
};

fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("GLFW: {}: {s}\n", .{ error_code, description });
}

var window: glfw.Window = undefined;
var gc: GraphicsContext = undefined;
var swapchain: Swapchain = undefined;
var render_pass: vk.RenderPass = undefined;
var descriptor_set_layout: vk.DescriptorSetLayout = undefined;
var pipeline_layout: vk.PipelineLayout = undefined;
var graphics_pipeline: vk.Pipeline = undefined;
var command_pool: vk.CommandPool = undefined;
var uniform_buffers: [max_frames_in_flight]vk.Buffer = undefined;
var uniform_buffers_memory: [max_frames_in_flight]vk.DeviceMemory = undefined;
var uniform_buffers_mapped: [max_frames_in_flight]*anyopaque = undefined;
var descriptor_pool: vk.DescriptorPool = undefined;
var descriptor_sets: [max_frames_in_flight]vk.DescriptorSet = undefined;
var frame_buffers: []vk.Framebuffer = undefined;
var command_buffers: []vk.CommandBuffer = undefined;
var vertex_buffer: vk.Buffer = undefined;
var vertex_buffer_memory: vk.DeviceMemory = undefined;
var index_buffer: vk.Buffer = undefined;
var index_buffer_memory: vk.DeviceMemory = undefined;
var start_time: i64 = undefined;

pub fn main() !void {
    start_time = std.time.milliTimestamp();

    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{})) {
        std.log.err("Failed to initialise GLFW: {?s}", .{ glfw.getErrorString() });
        std.process.exit(1);
    }
    defer glfw.terminate();

    var extent = vk.Extent2D { .width = 800, .height = 600 };
    window = glfw.Window.create(
        extent.width, extent.height, app_name, null, null, .{ .client_api = .no_api, }) 
        orelse {
            std.log.err("Failed to create GLFW window: {?s}", .{ glfw.getErrorString() });
            std.process.exit(1);
        };
    defer window.destroy();

    const allocator = std.heap.page_allocator;

    zmesh.init(allocator);
    defer zmesh.deinit();

    gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {?s}\n", .{gc.props.device_name});

    swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    try createRenderPass();
    defer gc.vkd.destroyRenderPass(gc.dev, render_pass, null);

    try createDescriptorSetLayout();
    defer gc.vkd.destroyDescriptorSetLayout(gc.dev, descriptor_set_layout, null);

    try createPipeline();
    defer destroyPipeline();

    try createFramebuffers(allocator);
    defer destroyFramebuffers(allocator);

    command_pool = try gc.vkd.createCommandPool(gc.dev, &.{
        .flags = .{},
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.vkd.destroyCommandPool(gc.dev, command_pool, null);

    try createVertexBuffer();
    defer destroyVertexBuffer();

    try createUniformBuffers();
    defer destroyUniformBuffers();

    try createDescriptorPool();
    defer gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);

    try createDescriptorSets();

    try createCommandBuffers(allocator);
    defer destroyCommandBuffers(allocator);

    while (!window.shouldClose()) {
        glfw.pollEvents();
        try drawFrame(allocator);
    }

    try gc.vkd.deviceWaitIdle(gc.dev);
}

const MappedBuffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
};

fn createBuffer(size: vk.DeviceSize, usage: vk.BufferUsageFlags, memory_flags: vk.MemoryPropertyFlags, buffer: *vk.Buffer, device_memory: *vk.DeviceMemory) !void {
    const create_info = vk.BufferCreateInfo {
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    };
    buffer.* = try gc.vkd.createBuffer(gc.dev, &create_info, null);
    //defer gc.vkd.destroyBuffer(gc.dev, buffer, null);

    const mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, buffer.*);
    device_memory.* = try gc.allocate(mem_reqs, memory_flags);
    //defer gc.vkd.freeMemory(gc.dev, memory, null);

    try gc.vkd.bindBufferMemory(gc.dev, buffer.*, device_memory.*, 0);
}

fn updateUniformBuffer(memory: *anyopaque) void {
    var current_time = std.time.milliTimestamp();
    var delta = current_time - start_time;
    var fdelta = @as(f32, @floatFromInt(delta)) / 1000.0;
    var ubo = UniformBufferObject{
        .model = zmath.rotationY(fdelta * std.math.degreesToRadians(f32, 90.0)),
        .view = zmath.lookAtLh(.{ 2.0, 2.0, 2.0, 1.0 }, .{ 0.0, 0.0, 0.0, 1.0 }, .{ 0.0, 1.0, 0.0, 0.0 }),
        .proj = zmath.perspectiveFovLh(std.math.degreesToRadians(f32, 45.0), @as(f32, @floatFromInt(swapchain.extent.width)) / @as(f32, @floatFromInt(swapchain.extent.height)), 0.1, 10.0)
    };
    var dst = @as([*]u8, @ptrCast(@alignCast(memory)))[0..@sizeOf(UniformBufferObject)];
    var src = std.mem.asBytes(&ubo);
    @memcpy(dst, src);
}

fn createDescriptorSetLayout() !void {
    const ubo_bindings = [1]vk.DescriptorSetLayoutBinding{ .{
        .binding = 0,
        .descriptor_type = .uniform_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex_bit = true },
        .p_immutable_samplers = null,
    }};

    const layout_info = vk.DescriptorSetLayoutCreateInfo{
        .binding_count = 1,
        .p_bindings = @ptrCast(&ubo_bindings),
    };

    descriptor_set_layout = try gc.vkd.createDescriptorSetLayout(gc.dev, &layout_info, null);
}

fn createVertexBuffer() !void {
    var staging_buffer: vk.Buffer = undefined;
    var staging_memory: vk.DeviceMemory = undefined;
    const buffer_size = @sizeOf(@TypeOf(vertices));
    try createBuffer(buffer_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, &staging_buffer, &staging_memory);
    defer gc.vkd.destroyBuffer(gc.dev, staging_buffer, null);
    defer gc.vkd.freeMemory(gc.dev, staging_memory, null);

    {
        const data = try gc.vkd.mapMemory(gc.dev, staging_memory, 0, buffer_size, .{});
        defer gc.vkd.unmapMemory(gc.dev, staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        for (vertices, 0..) |vertex, i| {
            gpu_vertices[i] = vertex;
        }
    }

    try createBuffer(buffer_size, .{ .transfer_dst_bit = true, .vertex_buffer_bit = true }, .{ .device_local_bit = true }, &vertex_buffer, &vertex_buffer_memory);
    try copyBuffer(vertex_buffer, staging_buffer, buffer_size);
}

fn destroyVertexBuffer() void {
    gc.vkd.destroyBuffer(gc.dev, vertex_buffer, null);
    gc.vkd.freeMemory(gc.dev, vertex_buffer_memory, null);
}

fn copyBuffer(dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf: vk.CommandBuffer = undefined;
    try gc.vkd.allocateCommandBuffers(gc.dev, &.{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    defer gc.vkd.freeCommandBuffers(gc.dev, command_pool, 1, @ptrCast(&cmdbuf));

    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    gc.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast(&region));

    try gc.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try gc.vkd.queueSubmit(gc.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try gc.vkd.queueWaitIdle(gc.graphics_queue.handle);
}

fn createCommandBuffers(allocator: Allocator) !void {
    command_buffers = try allocator.alloc(vk.CommandBuffer, frame_buffers.len);
    errdefer allocator.free(command_buffers);

    try gc.vkd.allocateCommandBuffers(gc.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = @truncate(command_buffers.len),
    }, command_buffers.ptr);
    errdefer gc.vkd.freeCommandBuffers(gc.dev, command_pool, @truncate(command_buffers.len), command_buffers.ptr);

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @as(f32, @floatFromInt(swapchain.extent.width)),
        .height = @as(f32, @floatFromInt(swapchain.extent.height)),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = swapchain.extent,
    };

    for (command_buffers, 0..) |cmdbuf, i| {
        try gc.vkd.beginCommandBuffer(cmdbuf, &.{
            .flags = .{},
            .p_inheritance_info = null,
        });

        gc.vkd.cmdSetViewport(cmdbuf, 0, 1, @as([*]const vk.Viewport, @ptrCast(&viewport)));
        gc.vkd.cmdSetScissor(cmdbuf, 0, 1, @as([*]const vk.Rect2D, @ptrCast(&scissor)));

        // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swapchain.extent,
        };

        gc.vkd.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = render_pass,
            .framebuffer = frame_buffers[i],
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @as([*]const vk.ClearValue, @ptrCast(&clear)),
        }, .@"inline");

        gc.vkd.cmdBindPipeline(cmdbuf, .graphics, graphics_pipeline);
        const offset = [_]vk.DeviceSize{0};
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @as([*]const vk.Buffer, @ptrCast(&vertex_buffer)), &offset);
        gc.vkd.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);

        gc.vkd.cmdEndRenderPass(cmdbuf);
        try gc.vkd.endCommandBuffer(cmdbuf);
    }
}

fn destroyCommandBuffers(allocator: Allocator) void {
    gc.vkd.freeCommandBuffers(gc.dev, command_pool, @truncate(command_buffers.len), command_buffers.ptr);
    allocator.free(command_buffers);
}

fn createFramebuffers(allocator: Allocator) !void {
    frame_buffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(frame_buffers);

    var i: usize = 0;
    errdefer for (frame_buffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (frame_buffers) |*fb| {
        fb.* = try gc.vkd.createFramebuffer(gc.dev, &vk.FramebufferCreateInfo{
            .flags = .{},
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }
}

fn destroyFramebuffers(allocator: Allocator) void {
    for (frame_buffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(frame_buffers);
}

fn createUniformBuffers() !void {
    for (0..max_frames_in_flight) |i| {
        try createBuffer(@sizeOf(@TypeOf(vertices)), 
            .{ 
                .uniform_buffer_bit = true 
            }, 
            .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            }, 
            &uniform_buffers[i], &uniform_buffers_memory[i]);

        // NOTE(bryan): Persistent mapping.
        var mapped_memory = try gc.vkd.mapMemory(gc.dev, uniform_buffers_memory[i], 0, @sizeOf(UniformBufferObject), .{});
        if (mapped_memory) |m| {
            uniform_buffers_mapped[i] = m;
        }
    }
}

fn destroyUniformBuffers() void {
    for (0..max_frames_in_flight) |i| {
        gc.vkd.destroyBuffer(gc.dev, uniform_buffers[i], null);
        gc.vkd.freeMemory(gc.dev, uniform_buffers_memory[i], null);
    }
}

fn createDescriptorPool() !void {
    var pool_size = vk.DescriptorPoolSize{
        .descriptor_count = max_frames_in_flight,
        .type = .uniform_buffer
    };
    var pool_info = vk.DescriptorPoolCreateInfo{
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&pool_size),
        .max_sets = max_frames_in_flight,
    };

    descriptor_pool = try gc.vkd.createDescriptorPool(gc.dev, &pool_info, null);
}

fn createDescriptorSets() !void {
    var layouts: [max_frames_in_flight]vk.DescriptorSetLayout = undefined;
    @memset(&layouts, descriptor_set_layout);
    var alloc_info = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = max_frames_in_flight,
        .p_set_layouts = &layouts,
    };
    try gc.vkd.allocateDescriptorSets(gc.dev, &alloc_info, &descriptor_sets);

    for (0..max_frames_in_flight) |i| {
        var buffer_info = vk.DescriptorBufferInfo{
            .buffer = uniform_buffers[i],
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        var descriptor_write = vk.WriteDescriptorSet{
            .dst_set = descriptor_sets[i],
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .p_buffer_info = @ptrCast(&buffer_info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };

        gc.vkd.updateDescriptorSets(gc.dev, 1, @ptrCast(&descriptor_write), 0, null);
    }
}

fn createRenderPass() !void {
    const color_attachment = vk.AttachmentDescription{
        .flags = .{},
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_resolve_attachments = null,
        .p_depth_stencil_attachment = null,
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };

    render_pass = try gc.vkd.createRenderPass(gc.dev, &vk.RenderPassCreateInfo{
        .flags = .{},
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 0,
        .p_dependencies = undefined,
    }, null);
}

fn createPipeline() !void {
    const pipeline_layout_info = vk.PipelineLayoutCreateInfo {
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&descriptor_set_layout),
        .push_constant_range_count = 0,
    };

    pipeline_layout = try gc.vkd.createPipelineLayout(gc.dev, &pipeline_layout_info, null);

    const vert = try gc.vkd.createShaderModule(gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = uber_vert.len,
        .p_code = @ptrCast(@alignCast(uber_vert)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, vert, null);

    const frag = try gc.vkd.createShaderModule(gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = triangle_frag.len,
        .p_code = @ptrCast(@alignCast(triangle_frag)),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, frag, null);

    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .flags = .{},
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
            .p_specialization_info = null,
        },
        .{
            .flags = .{},
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
            .p_specialization_info = null,
        },
    };

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .flags = .{},
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @as([*]const vk.VertexInputBindingDescription, @ptrCast(&Vertex.binding_description)),
        .vertex_attribute_description_count = Vertex.attribute_description.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .flags = .{},
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .flags = .{},
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .flags = .{},
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .flags = .{},
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .flags = .{},
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @as([*]const vk.PipelineColorBlendAttachmentState, @ptrCast(&pcbas)),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = 2,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    _ = try gc.vkd.createGraphicsPipelines(
        gc.dev,
        .null_handle,
        1,
        @as([*]const vk.GraphicsPipelineCreateInfo, @ptrCast(&gpci)),
        null,
        @as([*]vk.Pipeline, @ptrCast(&graphics_pipeline)),
    );
}

fn destroyPipeline() void {
    gc.vkd.destroyPipelineLayout(gc.dev, pipeline_layout, null);
    gc.vkd.destroyPipeline(gc.dev, graphics_pipeline, null);
}

fn drawFrame(allocator: Allocator) !void {
    const cmdbuf = command_buffers[swapchain.image_index];
    //updateUniformBuffer(uniform_buffers_mapped[swapchain.image_index % max_frames_in_flight]);

    const state = swapchain.present(cmdbuf) catch |err| switch (err) {
        error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
        else => |narrow| return narrow,
    };

    if (state == .suboptimal) {
        const size = window.getSize();
        swapchain.extent.width = @intCast(size.width);
        swapchain.extent.height = @intCast(size.height);
        try swapchain.recreate(swapchain.extent);

        destroyFramebuffers(allocator);
        try createFramebuffers(allocator);

        destroyCommandBuffers(allocator);
        try createCommandBuffers(allocator);
    }
}
