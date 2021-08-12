#include <Expand_Dims.hpp>
#define LOG_TAG "Expand_Dims"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Expand_Dims::Expand_Dims(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Expand_Dims::validate() {
    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank < 1) return false;

    return true;
}

std::shared_ptr<ngraph::Node> Expand_Dims::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto index = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);

    auto axes = createConstNode(ngraph::element::i32, {}, convertToVector(index));

    auto outputNode = std::make_shared<ngraph::opset3::Unsqueeze>(input, axes);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
