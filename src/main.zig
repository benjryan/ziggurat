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

const Error = error {
    AcquireNextImage,
    PresentImage
};

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

fn framebufferSizeCallback(window: glfw.Window, width: u32, height: u32) void {
    _ = height;
    _ = width;
    if (window.getUserPointer(App)) |app| {
        app.framebufferResized = true;
    }
}

// FIXME(bryan): Globals are bad specifically because they're never in scope when debugging so we can't see their values.

const App = struct {
    allocator: Allocator = undefined,
    window: glfw.Window = undefined,
    gc: GraphicsContext = undefined,
    swapchain: Swapchain = undefined,
    render_pass: vk.RenderPass = undefined,
    descriptor_set_layout: vk.DescriptorSetLayout = undefined,
    pipeline_layout: vk.PipelineLayout = undefined,
    graphics_pipeline: vk.Pipeline = undefined,
    command_pool: vk.CommandPool = undefined,
    uniform_buffers: [max_frames_in_flight]vk.Buffer = undefined,
    uniform_buffers_memory: [max_frames_in_flight]vk.DeviceMemory = undefined,
    uniform_buffers_mapped: [max_frames_in_flight]*anyopaque = undefined,
    descriptor_pool: vk.DescriptorPool = undefined,
    descriptor_sets: [max_frames_in_flight]vk.DescriptorSet = undefined,
    frame_buffers: []vk.Framebuffer = undefined,
    command_buffers: [max_frames_in_flight]vk.CommandBuffer = undefined,
    vertex_buffer: vk.Buffer = undefined,
    vertex_buffer_memory: vk.DeviceMemory = undefined,
    index_buffer: vk.Buffer = undefined,
    index_buffer_memory: vk.DeviceMemory = undefined,
    image_available_semaphores: [max_frames_in_flight]vk.Semaphore = undefined,
    render_finished_semaphores: [max_frames_in_flight]vk.Semaphore = undefined,
    in_flight_fences: [max_frames_in_flight]vk.Fence = undefined,
    current_frame: u32 = 0,
    framebufferResized: bool = false,
    start_time: i64 = undefined,
};

pub fn main() !void {
    var app = App{};
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    app.allocator = gpa.allocator();

    app.start_time = std.time.milliTimestamp();

    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{})) {
        std.log.err("Failed to initialise GLFW: {?s}", .{ glfw.getErrorString() });
        std.process.exit(1);
    }
    defer glfw.terminate();

    var extent = vk.Extent2D { .width = 800, .height = 600 };
    app.window = glfw.Window.create(
        extent.width, extent.height, app_name, null, null, .{ .client_api = .no_api, }) 
        orelse {
            std.log.err("Failed to create GLFW window: {?s}", .{ glfw.getErrorString() });
            std.process.exit(1);
        };
    defer app.window.destroy();
    app.window.setUserPointer(&app);
    app.window.setFramebufferSizeCallback(framebufferSizeCallback);

    zmesh.init(app.allocator);
    defer zmesh.deinit();

    app.gc = try GraphicsContext.init(app.allocator, app_name, app.window);
    defer app.gc.deinit();

    std.debug.print("Using device: {?s}\n", .{app.gc.props.device_name});

    app.swapchain = try Swapchain.init(&app.gc, app.allocator, extent);
    defer app.swapchain.deinit();

    try createRenderPass(&app);
    defer app.gc.vkd.destroyRenderPass(app.gc.dev, app.render_pass, null);

    try createDescriptorSetLayout(&app);
    defer app.gc.vkd.destroyDescriptorSetLayout(app.gc.dev, app.descriptor_set_layout, null);

    try createPipeline(&app);
    defer destroyPipeline(&app);

    try createFramebuffers(&app);
    defer destroyFramebuffers(&app);

    app.command_pool = try app.gc.vkd.createCommandPool(app.gc.dev, &.{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = app.gc.graphics_queue.family,
    }, null);
    defer app.gc.vkd.destroyCommandPool(app.gc.dev, app.command_pool, null);

    try createVertexBuffer(&app);
    defer destroyVertexBuffer(&app);

    try createUniformBuffers(&app);
    defer destroyUniformBuffers(&app);

    try createDescriptorPool(&app);
    defer app.gc.vkd.destroyDescriptorPool(app.gc.dev, app.descriptor_pool, null);

    try createDescriptorSets(&app);

    try createCommandBuffers(&app);
    defer destroyCommandBuffers(&app);

    try createSyncObjects(&app);
    defer destroySyncObjects(&app);

    while (!app.window.shouldClose()) {
        glfw.pollEvents();
        try drawFrame(&app);
    }

    try app.gc.vkd.deviceWaitIdle(app.gc.dev);
}

const MappedBuffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
};

fn createBuffer(app: *App, size: vk.DeviceSize, usage: vk.BufferUsageFlags, memory_flags: vk.MemoryPropertyFlags, buffer: *vk.Buffer, device_memory: *vk.DeviceMemory) !void {
    const create_info = vk.BufferCreateInfo {
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    };
    buffer.* = try app.gc.vkd.createBuffer(app.gc.dev, &create_info, null);
    //defer gc.vkd.destroyBuffer(gc.dev, buffer, null);

    const mem_reqs = app.gc.vkd.getBufferMemoryRequirements(app.gc.dev, buffer.*);
    device_memory.* = try app.gc.allocate(mem_reqs, memory_flags);
    //defer gc.vkd.freeMemory(gc.dev, memory, null);

    try app.gc.vkd.bindBufferMemory(app.gc.dev, buffer.*, device_memory.*, 0);
}

fn updateUniformBuffer(app: *App) void {
    var current_time = std.time.milliTimestamp();
    var delta = current_time - app.start_time;
    var fdelta = @as(f32, @floatFromInt(delta)) / 1000.0;

    var ubo = UniformBufferObject{
        .model = zmath.rotationY(fdelta * std.math.degreesToRadians(f32, 90.0)),
        .view = zmath.lookAtLh(.{ 2.0, 2.0, 2.0, 1.0 }, .{ 0.0, 0.0, 0.0, 1.0 }, .{ 0.0, 1.0, 0.0, 0.0 }),
        .proj = zmath.perspectiveFovLh(std.math.degreesToRadians(f32, 45.0), @as(f32, @floatFromInt(app.swapchain.extent.width)) / @as(f32, @floatFromInt(app.swapchain.extent.height)), 0.1, 10.0)
    };

    var mem: *UniformBufferObject = @ptrCast(@alignCast(app.uniform_buffers_mapped[app.current_frame]));
    mem.* = ubo;
}

fn createDescriptorSetLayout(app: *App) !void {
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

    app.descriptor_set_layout = try app.gc.vkd.createDescriptorSetLayout(app.gc.dev, &layout_info, null);
}

fn createVertexBuffer(app: *App) !void {
    var staging_buffer: vk.Buffer = undefined;
    var staging_memory: vk.DeviceMemory = undefined;
    const buffer_size = @sizeOf(@TypeOf(vertices));
    try createBuffer(app, buffer_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, &staging_buffer, &staging_memory);
    defer app.gc.vkd.destroyBuffer(app.gc.dev, staging_buffer, null);
    defer app.gc.vkd.freeMemory(app.gc.dev, staging_memory, null);

    {
        const data = try app.gc.vkd.mapMemory(app.gc.dev, staging_memory, 0, buffer_size, .{});
        defer app.gc.vkd.unmapMemory(app.gc.dev, staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        for (vertices, 0..) |vertex, i| {
            gpu_vertices[i] = vertex;
        }
    }

    try createBuffer(app, buffer_size, .{ .transfer_dst_bit = true, .vertex_buffer_bit = true }, .{ .device_local_bit = true }, &app.vertex_buffer, &app.vertex_buffer_memory);
    try copyBuffer(app, app.vertex_buffer, staging_buffer, buffer_size);
}

fn destroyVertexBuffer(app: *App) void {
    app.gc.vkd.destroyBuffer(app.gc.dev, app.vertex_buffer, null);
    app.gc.vkd.freeMemory(app.gc.dev, app.vertex_buffer_memory, null);
}

fn copyBuffer(app: *App, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf: vk.CommandBuffer = undefined;
    try app.gc.vkd.allocateCommandBuffers(app.gc.dev, &.{
        .command_pool = app.command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    defer app.gc.vkd.freeCommandBuffers(app.gc.dev, app.command_pool, 1, @ptrCast(&cmdbuf));

    try app.gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    app.gc.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast(&region));

    try app.gc.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try app.gc.vkd.queueSubmit(app.gc.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try app.gc.vkd.queueWaitIdle(app.gc.graphics_queue.handle);
}

fn createCommandBuffers(app: *App) !void {
    try app.gc.vkd.allocateCommandBuffers(app.gc.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = app.command_pool,
        .level = .primary,
        .command_buffer_count = max_frames_in_flight,
    }, @ptrCast(&app.command_buffers));
    errdefer app.gc.vkd.freeCommandBuffers(app.gc.dev, app.command_pool, max_frames_in_flight, @ptrCast(&app.command_buffers));
}

fn recordCommandBuffer(app: *App, command_buffer: vk.CommandBuffer, image_index: u32) !void {
    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @as(f32, @floatFromInt(app.swapchain.extent.width)),
        .height = @as(f32, @floatFromInt(app.swapchain.extent.height)),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = app.swapchain.extent,
    };

    try app.gc.vkd.beginCommandBuffer(command_buffer, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

    const render_area = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = app.swapchain.extent,
    };

    const render_pass_info = vk.RenderPassBeginInfo{
        .render_pass = app.render_pass,
        .framebuffer = app.frame_buffers[image_index],
        .render_area = render_area,
        .clear_value_count = 1,
        .p_clear_values = @ptrCast(&clear),
    };

    app.gc.vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");
    app.gc.vkd.cmdBindPipeline(command_buffer, .graphics, app.graphics_pipeline);
    app.gc.vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&viewport));
    app.gc.vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&scissor));

    const offsets = [_]vk.DeviceSize{0};
    app.gc.vkd.cmdBindVertexBuffers(command_buffer, 0, 1, @ptrCast(&app.vertex_buffer), &offsets);
    //gc.vkd.cmdBindIndexBuffer(command_buffer, index_buffer, 0, .uint16);
    app.gc.vkd.cmdBindDescriptorSets(command_buffer, .graphics, app.pipeline_layout, 0, 1, @ptrCast(&app.descriptor_sets[app.current_frame]), 0, null);
    app.gc.vkd.cmdDraw(command_buffer, vertices.len, 1, 0, 0);

    app.gc.vkd.cmdEndRenderPass(command_buffer);
    try app.gc.vkd.endCommandBuffer(command_buffer);
}

fn destroyCommandBuffers(app: *App) void {
    app.gc.vkd.freeCommandBuffers(app.gc.dev, app.command_pool, @truncate(app.command_buffers.len), @ptrCast(&app.command_buffers));
}

fn createFramebuffers(app: *App) !void {
    app.frame_buffers = try app.allocator.alloc(vk.Framebuffer, app.swapchain.swap_images.len);
    errdefer app.allocator.free(app.frame_buffers);

    var i: usize = 0;
    errdefer for (app.frame_buffers[0..i]) |fb| app.gc.vkd.destroyFramebuffer(app.gc.dev, fb, null);

    for (app.frame_buffers) |*fb| {
        fb.* = try app.gc.vkd.createFramebuffer(app.gc.dev, &vk.FramebufferCreateInfo{
            .flags = .{},
            .render_pass = app.render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&app.swapchain.swap_images[i].view),
            .width = app.swapchain.extent.width,
            .height = app.swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }
}

fn destroyFramebuffers(app: *App) void {
    for (app.frame_buffers) |fb| app.gc.vkd.destroyFramebuffer(app.gc.dev, fb, null);
    app.allocator.free(app.frame_buffers);
}

fn createUniformBuffers(app: *App) !void {
    for (0..max_frames_in_flight) |i| {
        const buffer_size = @sizeOf(UniformBufferObject);
        try createBuffer(app, buffer_size, 
            .{ 
                .uniform_buffer_bit = true 
            }, 
            .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            }, 
            &app.uniform_buffers[i], &app.uniform_buffers_memory[i]);

        // NOTE(bryan): Persistent mapping.
        var mapped_memory = try app.gc.vkd.mapMemory(app.gc.dev, app.uniform_buffers_memory[i], 0, buffer_size, .{});
        std.debug.assert(mapped_memory != null);
        if (mapped_memory) |m| {
            app.uniform_buffers_mapped[i] = m;
        }
    }
}

fn destroyUniformBuffers(app: *App) void {
    for (0..max_frames_in_flight) |i| {
        app.gc.vkd.destroyBuffer(app.gc.dev, app.uniform_buffers[i], null);
        app.gc.vkd.freeMemory(app.gc.dev, app.uniform_buffers_memory[i], null);
    }
}

fn createDescriptorPool(app: *App) !void {
    var pool_size = vk.DescriptorPoolSize{
        .descriptor_count = max_frames_in_flight,
        .type = .uniform_buffer
    };
    var pool_info = vk.DescriptorPoolCreateInfo{
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&pool_size),
        .max_sets = max_frames_in_flight,
    };

    app.descriptor_pool = try app.gc.vkd.createDescriptorPool(app.gc.dev, &pool_info, null);
}

fn createDescriptorSets(app: *App) !void {
    var layouts: [max_frames_in_flight]vk.DescriptorSetLayout = undefined;
    @memset(layouts[0..max_frames_in_flight], app.descriptor_set_layout);
    var alloc_info = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = app.descriptor_pool,
        .descriptor_set_count = max_frames_in_flight,
        .p_set_layouts = &layouts,
    };
    try app.gc.vkd.allocateDescriptorSets(app.gc.dev, &alloc_info, &app.descriptor_sets);

    for (0..max_frames_in_flight) |i| {
        var buffer_info = vk.DescriptorBufferInfo{
            .buffer = app.uniform_buffers[i],
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        var descriptor_write = vk.WriteDescriptorSet{
            .dst_set = app.descriptor_sets[i],
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .p_buffer_info = @ptrCast(&buffer_info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };

        app.gc.vkd.updateDescriptorSets(app.gc.dev, 1, @ptrCast(&descriptor_write), 0, null);
    }
}

fn createRenderPass(app: *App) !void {
    const color_attachment = vk.AttachmentDescription{
        .flags = .{},
        .format = app.swapchain.surface_format.format,
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

    app.render_pass = try app.gc.vkd.createRenderPass(app.gc.dev, &vk.RenderPassCreateInfo{
        .flags = .{},
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 0,
        .p_dependencies = undefined,
    }, null);
}

fn createPipeline(app: *App) !void {
    const pipeline_layout_info = vk.PipelineLayoutCreateInfo {
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&app.descriptor_set_layout),
    };

    app.pipeline_layout = try app.gc.vkd.createPipelineLayout(app.gc.dev, &pipeline_layout_info, null);

    const vert = try app.gc.vkd.createShaderModule(app.gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = uber_vert.len,
        .p_code = @ptrCast(@alignCast(uber_vert)),
    }, null);
    defer app.gc.vkd.destroyShaderModule(app.gc.dev, vert, null);

    const frag = try app.gc.vkd.createShaderModule(app.gc.dev, &vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = triangle_frag.len,
        .p_code = @ptrCast(@alignCast(triangle_frag)),
    }, null);
    defer app.gc.vkd.destroyShaderModule(app.gc.dev, frag, null);

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
        .layout = app.pipeline_layout,
        .render_pass = app.render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    _ = try app.gc.vkd.createGraphicsPipelines(
        app.gc.dev,
        .null_handle,
        1,
        @as([*]const vk.GraphicsPipelineCreateInfo, @ptrCast(&gpci)),
        null,
        @as([*]vk.Pipeline, @ptrCast(&app.graphics_pipeline)),
    );
}

fn destroyPipeline(app: *App) void {
    app.gc.vkd.destroyPipelineLayout(app.gc.dev, app.pipeline_layout, null);
    app.gc.vkd.destroyPipeline(app.gc.dev, app.graphics_pipeline, null);
}

fn drawFrame(app: *App) !void {
    _ = try app.gc.vkd.waitForFences(app.gc.dev, 1, @ptrCast(&app.in_flight_fences[app.current_frame]), vk.TRUE, std.math.maxInt(u64));

    var acquire_result = try app.gc.vkd.acquireNextImageKHR(app.gc.dev, app.swapchain.handle, std.math.maxInt(u64), app.image_available_semaphores[app.current_frame], .null_handle);
    var image_index = acquire_result.image_index;

    if (acquire_result.result == .error_out_of_date_khr) {
        try recreateSwapChain(app);
        return;
    } else if (acquire_result.result != .success and acquire_result.result != .suboptimal_khr) {
        std.log.err("Failed to acquire swap chain image!", .{});
        return error.AcquireNextImage;
    }

    updateUniformBuffer(app);
    try app.gc.vkd.resetFences(app.gc.dev, 1, @ptrCast(&app.in_flight_fences[app.current_frame]));
    try app.gc.vkd.resetCommandBuffer(app.command_buffers[app.current_frame], .{});
    try recordCommandBuffer(app, app.command_buffers[app.current_frame], image_index);

    var wait_semaphores = [_]vk.Semaphore{ app.image_available_semaphores[app.current_frame] };
    var wait_stages = [_]vk.PipelineStageFlags{ vk.PipelineStageFlags{ .color_attachment_output_bit = true }};
    var signal_semaphores = [_]vk.Semaphore{ app.render_finished_semaphores[app.current_frame] };
    var submit_info = vk.SubmitInfo{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&wait_semaphores),
        .p_wait_dst_stage_mask = @ptrCast(&wait_stages),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&app.command_buffers[app.current_frame]),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = @ptrCast(&signal_semaphores),
    };

    try app.gc.vkd.queueSubmit(app.gc.graphics_queue.handle, 1, @ptrCast(&submit_info), app.in_flight_fences[app.current_frame]);

    var swapchains = [_]vk.SwapchainKHR { app.swapchain.handle };
    var present_info = vk.PresentInfoKHR{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&signal_semaphores),
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&swapchains),
        .p_image_indices = @ptrCast(&image_index),
    };
    var present_result = try app.gc.vkd.queuePresentKHR(app.gc.present_queue.handle, &present_info);
    if (present_result == .error_out_of_date_khr or
        present_result == .suboptimal_khr or
        app.framebufferResized) {
        app.framebufferResized = false;
        try recreateSwapChain(app);
    } else if (present_result != .success) {
        std.log.err("Failed to present swapchain image!", .{});
        return error.PresentImage;
    }

    app.current_frame = (app.current_frame + 1) % max_frames_in_flight;
}

fn recreateSwapChain(app: *App) !void {
    const size = app.window.getSize();
    app.swapchain.extent.width = @intCast(size.width);
    app.swapchain.extent.height = @intCast(size.height);
    try app.swapchain.recreate(app.swapchain.extent);

    destroyFramebuffers(app);
    try createFramebuffers(app);

    destroyCommandBuffers(app);
    try createCommandBuffers(app);
}

//fn drawFrame(allocator: Allocator) !void {
//    const cmdbuf = command_buffers[swapchain.image_index];
//    updateUniformBuffer(uniform_buffers_mapped[swapchain.image_index % max_frames_in_flight]);
//
//    const state = swapchain.present(cmdbuf) catch |err| switch (err) {
//        error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
//        else => |narrow| return narrow,
//    };
//
//    if (state == .suboptimal) {
//        const size = window.getSize();
//        swapchain.extent.width = @intCast(size.width);
//        swapchain.extent.height = @intCast(size.height);
//        try swapchain.recreate(swapchain.extent);
//
//        destroyFramebuffers(allocator);
//        try createFramebuffers(allocator);
//
//        destroyCommandBuffers(allocator);
//        try createCommandBuffers(allocator);
//    }
//}

fn createSyncObjects(app: *App) !void {
    for (0..max_frames_in_flight) |i| {
        var semaphore_info = vk.SemaphoreCreateInfo {};
        app.image_available_semaphores[i] = try app.gc.vkd.createSemaphore(app.gc.dev, &semaphore_info, null);
        app.render_finished_semaphores[i] = try app.gc.vkd.createSemaphore(app.gc.dev, &semaphore_info, null);

        var fence_info = vk.FenceCreateInfo {
            .flags = .{ .signaled_bit = true }
        };
        app.in_flight_fences[i] = try app.gc.vkd.createFence(app.gc.dev, &fence_info, null);
    }
}

fn destroySyncObjects(app: *App) void {
    for (0..max_frames_in_flight) |i| {
        app.gc.vkd.destroySemaphore(app.gc.dev, app.image_available_semaphores[i], null);
        app.gc.vkd.destroySemaphore(app.gc.dev, app.render_finished_semaphores[i], null);
        app.gc.vkd.destroyFence(app.gc.dev, app.in_flight_fences[i], null);
    }
}
