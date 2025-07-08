#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::averagepool {

struct Test::Attributes {
    std::vector<int32_t> kernel_size;
    std::vector<int32_t> stride;
    std::vector<int32_t> padding;
    bool ceil_mode;
    bool count_include_pad;
    int32_t divisor_override; // -1 表示未设置

    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output; // 期望输出
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    // 解析参数
    auto kernel_size_data = attributes["kernel_size"];
    test->_attributes->kernel_size.resize(kernel_size_data.size() / sizeof(int32_t));
    std::memcpy(test->_attributes->kernel_size.data(), kernel_size_data.data(), kernel_size_data.size());

    if (attributes.find("stride") != attributes.end()) {
        auto stride_data = attributes["stride"];
        test->_attributes->stride.resize(stride_data.size() / sizeof(int32_t));
        std::memcpy(test->_attributes->stride.data(), stride_data.data(), stride_data.size());
    }

    auto padding_data = attributes["padding"];
    test->_attributes->padding.resize(padding_data.size() / sizeof(int32_t));
    std::memcpy(test->_attributes->padding.data(), padding_data.data(), padding_data.size());

    test->_attributes->ceil_mode = *reinterpret_cast<bool *>(attributes["ceil_mode"].data());
    test->_attributes->count_include_pad = *reinterpret_cast<bool *>(attributes["count_include_pad"].data());

    test->_attributes->divisor_override = -1; // 默认值
    if (attributes.find("divisor_override") != attributes.end()) {
        test->_attributes->divisor_override = *reinterpret_cast<int32_t *>(attributes["divisor_override"].data());
    }

    // 设置张量
    test->_attributes->input = tensors["input"];
    test->_attributes->output = tensors["output"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    // 创建操作描述符（假设有 infiniopCreateAveragePoolDescriptor）
    infiniopAveragePoolDescriptor_t op_desc;
    auto input = _attributes->input->to(device, device_id);
    auto output_shape = _attributes->output->shape();

    // 创建输出张量
    auto actual_output = std::make_shared<Tensor>(
        output_shape,
        _attributes->output->ggml_type(),
        device, device_id);

    CHECK_OR(infiniopCreateAveragePoolDescriptor(
                 handle, &op_desc,
                 actual_output->desc(),
                 input->desc(),
                 _attributes->kernel_size.data(),
                 _attributes->stride.empty() ? nullptr : _attributes->stride.data(),
                 _attributes->padding.data(),
                 _attributes->ceil_mode,
                 _attributes->count_include_pad,
                 _attributes->divisor_override == -1 ? nullptr : &_attributes->divisor_override),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create averagepool descriptor."));

    // 获取工作空间大小
    size_t workspace_size;
    CHECK_OR(infiniopGetAveragePoolWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    // 执行操作
    CHECK_OR(infiniopAveragePool(op_desc, workspace, workspace_size,
                                 actual_output->data(),
                                 input->data(),
                                 nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // 验证结果
    try {
        allClose(actual_output, _attributes->output, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopAveragePool(op_desc, workspace, workspace_size,
                                actual_output->data(),
                                input->data(),
                                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "output"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- kernel_size: [";
    for (size_t i = 0; i < _attributes->kernel_size.size(); ++i) {
        oss << _attributes->kernel_size[i];
        if (i < _attributes->kernel_size.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]" << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::averagepool