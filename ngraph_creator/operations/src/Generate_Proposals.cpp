#include <Generate_Proposals.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Generate_Proposals::Generate_Proposals(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Generate_Proposals::validate() {
    ALOGV("%s Entering", __func__);


    return true;
}

std::shared_ptr<ngraph::Node> Generate_Proposals::createNode() {
    ALOGV("%s Entering", __func__);

    bool useNchw = false;

    // 4-D Tensor specifying the score of each anchor at each location
    auto score_anchor = getInputNode(0);

    // 4-D Tensor specifying the bounding box deltas
    auto bbox_deltas = getInputNode(1);

    // 2-D Tensor of shape [num_anchors, 4], shape of each predefined anchor, with format [x1, y1, x2, y2].
    auto pre_anchor = getInputNode(2);

    // 2-D Tensor of shape [batches, 2], specifying the size of each image in the batch, with format [image_height, image_width]
    auto size_image = getInputNode(3);

    // ratio from the height of original image to the height of feature map.
    auto height_ratio = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 4);

    // ratio from the width of original image to the width of feature map.
    auto width_ratio = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex,  5);

    // maximum number of boxes before going into the hard NMS algorithm.
    // Boxes with the lowest scores are discarded to meet the limit. Set to a non-positive value for unlimited number.
    auto pre_max_boxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex,  6);

    // maximum number of boxes returning from the hard NMS algorithm.
    // Boxes with the lowest scores are discarded to meet the limit. Set to a non-positive value for unlimited number.
    auto post_max_boxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex,  7);

    // IoU threshold for hard NMS.
    auto iou_thres = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex,  8);

    // min_size. Boxes with height or width lower than the absolute threshold are filtered out.
    auto min_size = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex,  9);

    // set to true to specify NCHW data layout for input0 and input1. Set to false for NHWC.
    auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);

    if (layout) useNchw = true;

    if (!useNchw)  {  // No conversion needed if useNchw set
        score_anchor = transpose(NHWC_NCHW, score_anchor);
        bbox_deltas = transpose(NHWC_NCHW, bbox_deltas);
    }

    auto class_probs = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1024, 2, 128, 128});
    auto class_logits = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1024, 4, 128, 128});
    auto image_shape = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{4});

    ngraph::op::ProposalAttrs attrs;
    attrs.base_size = 224;
    attrs.pre_nms_topn = 100;
    attrs.post_nms_topn = 110;
    attrs.nms_thresh = 0.12f;
    attrs.feat_stride = 2;
    attrs.min_size = 10;
    attrs.ratio = std::vector<float>{1.44f, 0.66f};
    attrs.scale = std::vector<float>{2.25f, 1.83f};
    attrs.clip_before_nms = true;
    attrs.clip_after_nms = true;
    attrs.normalize = false;
    attrs.box_size_scale = 2.f;
    attrs.box_coordinate_scale = 4.55f;
    attrs.framework = std::string{"nGraph"};

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Proposal>(
        class_probs, class_logits, image_shape, attrs);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    ALOGV("%s PASSED", __func__);
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
