#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include <doctest/doctest.h>

#include "model/ffm.h"
#include "model/fm.h"
#include "model/lr.h"
#include "utils/types.h"

namespace ftrl {

constexpr int n_fields = 4, n_feats = 50, n_factors = 4;

TEST_CASE("models") {
  config_options args;
  args.n_fields = n_fields;
  args.n_feats = n_feats;
  args.n_factors = n_factors;

  SUBCASE("Logistic Regression") {
    args.model_type = "LR";
    LR model{args};
    CHECK_EQ(model.model_type, ModelType::LR);
    CHECK_EQ(model.lin_w.size(), n_feats);
    const feat_vec invalid_sample = {{1, -1, 3}, {1, 0, 1}, {1, 100, 0}};
    model.remove_out_range(const_cast<feat_vec &>(invalid_sample));
    CHECK_EQ(invalid_sample.size(), 1);
  }

  SUBCASE("FM") {
    args.model_type = "FM";
    FM model{args};
    CHECK_EQ(model.model_type, ModelType::FM);
    CHECK_EQ(model.lin_w.size(), n_feats);
    CHECK_EQ(model.vec_w.size(), n_feats);
    CHECK_EQ(model.vec_w[0].size(), n_factors);
  }

  SUBCASE("FFM") {
    args.model_type = "FFM";
    FFM model{args};
    CHECK_EQ(model.model_type, ModelType::FFM);
    CHECK_EQ(model.vec_w[0].size(), n_fields * n_factors);
    const feat_vec invalid_sample = {{1, -1, 3}, {44, 0, 1}, {1, 100, 0}};
    model.remove_out_range(const_cast<feat_vec &>(invalid_sample));
    CHECK(invalid_sample.empty());
  }
}

}  // namespace ftrl
