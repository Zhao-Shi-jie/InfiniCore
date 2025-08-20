#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::cross_entropy_loss {

struct Test::Attributes {
    // 输入张量
    std::shared_ptr<Tensor> logits;
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> expected_output;

    // 可选属性，保留用于验证
    bool has_weight;
    std::shared_ptr<Tensor> weight;
    int reduction;
    int ignore_index;
    float label_smoothing;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    std::cout << "DEBUG: cross_entropy_loss::Test::build called" << std::endl;

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->logits = tensors["logits"];
    test->_attributes->target = tensors["target"];
    test->_attributes->expected_output = tensors["output"];

    // weight
    if (tensors.find("weight") != tensors.end()) {
        test->_attributes->weight = tensors["weight"];
        test->_attributes->has_weight = true;
    } else {
        test->_attributes->has_weight = false;
    }

    // reduction
    if (attributes.find("reduction") != attributes.end() && attributes["reduction"].size() == sizeof(int)) {
        test->_attributes->reduction = *reinterpret_cast<const int*>(attributes["reduction"].data());
    } else {
        test->_attributes->reduction = 1; // default: mean
    }

    // ignore_index
    if (attributes.find("ignore_index") != attributes.end() && attributes["ignore_index"].size() == sizeof(int)) {
        test->_attributes->ignore_index = *reinterpret_cast<const int*>(attributes["ignore_index"].data());
    } else {
        test->_attributes->ignore_index = -100;
    }

    // label_smoothing
    if (attributes.find("label_smoothing") != attributes.end() && attributes["label_smoothing"].size() == sizeof(float)) {
        test->_attributes->label_smoothing = *reinterpret_cast<const float*>(attributes["label_smoothing"].data());
    } else {
        test->_attributes->label_smoothing = 0.0f;
    }

    std::cout << "DEBUG: build finished" << std::endl;
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopCrossEntropyLossDescriptor_t op_desc;

    auto logits = _attributes->logits->to(device, device_id);
    auto target = _attributes->target->to(device, device_id);
    auto expected_output = _attributes->expected_output;

    auto output_shape = expected_output->shape();
    size_t output_size = 1;
    for (auto dim : output_shape) output_size *= dim;
    output_size *= ggmlTypeSize(logits->ggml_type());

    auto output_memory = std::make_shared<Memory>(output_size, device, device_id);
    std::vector<ptrdiff_t> output_strides(output_shape.size());
    if (!output_shape.empty()) {
        output_strides.back() = 1;
        for (int i = output_shape.size() - 2; i >= 0; i--) {
            output_strides[i] = output_strides[i+1] * output_shape[i+1];
        }
    }

    auto actual_output = std::make_shared<Tensor>(
        output_memory, 0, output_shape, output_strides, logits->ggml_type()
    );

    std::cout << "DEBUG: Creating cross_entropy_loss descriptor..." << std::endl;
    CHECK_OR(infiniopCreateCrossEntropyLossDescriptor(
                 handle, &op_desc,
                 actual_output->desc(),
                 logits->desc(),
                 target->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create cross_entropy_loss descriptor."));

    // --- 获取 workspace 大小 ---
    size_t workspace_size = 0;
    CHECK_OR(infiniopGetCrossEntropyLossWorkspaceSize(op_desc, &workspace_size),
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void* workspace_ptr = nullptr;
    std::shared_ptr<Memory> workspace_memory;
    if (workspace_size > 0) {
        workspace_memory = std::make_shared<Memory>(workspace_size, device, device_id);
        workspace_ptr = workspace_memory->ptr();  // <-- 这里根据 Memory 类接口改
    }

    // 执行算子
    CHECK_OR(infiniopCrossEntropyLoss(
                op_desc,
                workspace_ptr,
                workspace_size,
                actual_output->data(),  // loss
                logits->data(),
                target->data(),
                nullptr),               // stream
            return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during cross_entropy_loss execution."));


    // 验证结果
    try {
        allClose(actual_output, expected_output, _rtol, _atol);
    } catch (const std::exception &e) {
        infiniopDestroyCrossEntropyLossDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopCrossEntropyLoss(
                op_desc,
                workspace_ptr,
                workspace_size,
                actual_output->data(),
                logits->data(),
                target->data(),
                nullptr);
        }, warm_ups, iterations);

    infiniopDestroyCrossEntropyLossDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"reduction", "ignore_index", "label_smoothing"};
}

std::vector<std::string> Test::tensor_names() {
    return {"logits", "target", "output", "weight"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- logits: " << _attributes->logits->info() << std::endl;
    oss << "- target: " << _attributes->target->info() << std::endl;
    oss << "- expected_output: " << _attributes->expected_output->info() << std::endl;
    oss << "- has_weight: " << (_attributes->has_weight ? "true" : "false") << std::endl;
    oss << "- reduction: " << _attributes->reduction << std::endl;
    oss << "- ignore_index: " << _attributes->ignore_index << std::endl;
    oss << "- label_smoothing: " << _attributes->label_smoothing << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::cross_entropy_loss
