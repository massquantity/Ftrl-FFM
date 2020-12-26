#ifndef FTRL_FFM_FTRL_TRAINER_H
#define FTRL_FFM_FTRL_TRAINER_H

#include <vector>
#include <memory>
#include <tuple>

#include "ftrl_model.h"
#include "../process/sample.h"
#include "../threading/pc_task.h"
#include "../utils/cmd_option.h"

namespace ftrl {

class ftrl_trainer : public pc_task {
public:
  explicit ftrl_trainer(const trainer_option &opt);
  void run_task(std::vector<std::string> &dataBuffer, int t) override;
  bool loadModel(std::ifstream &ifs);
  void outputModel(std::ofstream &ofs);
  std::shared_ptr<ftrl_model> pModel;

private:
  void train(int y, const std::vector<std::tuple<int, std::string, double>> &x);
  double w_alpha, w_beta, w_l1, w_l2;
  bool k0, k1;
};

ftrl_trainer::ftrl_trainer(const trainer_option &opt): pc_task(opt.thread_num, opt.cmd),
    w_alpha(opt.w_alpha), w_beta(opt.w_beta), w_l1(opt.w_l1), w_l2(opt.w_l2),
    k0(opt.k0), k1(opt.k1) {
  pModel = std::make_shared<ftrl_model>(
      opt.init_mean, opt.init_stddev, opt.n_factors, opt.n_fields);
}

void ftrl_trainer::run_task(std::vector<std::string> &dataBuffer, int t) {
  for (const auto &d : dataBuffer) {
    sample singleSample(d);
    train(singleSample.y, singleSample.x);
  }
}

bool ftrl_trainer::loadModel(std::ifstream &ifs) {
  return pModel->loadModel(ifs);
}

void ftrl_trainer::outputModel(std::ofstream &ofs) {
  return pModel->outputModel(ofs);
}

void ftrl_trainer::train(int y, const std::vector<std::tuple<int, std::string, double>> &x) {
  size_t xLen = x.size();
  std::vector<std::shared_ptr<ftrl_model_unit>> params(xLen + 1);
  for (int i = 0; i < xLen; i++) {
    int field_id = std::get<0>(x[i]);
    const std::string &index = std::get<1>(x[i]);
    params[i] = pModel->getOrInitWeight(field_id, index);
  }
  params[xLen] = pModel->getOrInitBias();

  for (int i = 0; i <= xLen; i++) {  // todo convert to function
    if ((i < xLen && k1) || (i == xLen && k0)) {
      ftrl_model_unit &mu = *params[i];
      std::unique_lock<std::mutex> lck(mu.mtx);
      if (fabs(mu.w_zi) <= w_l1) {
        mu.wi = 0.0;
      } else {
        mu.wi = -1 * (mu.w_zi - utils::sgn(mu.w_zi) * w_l1) /
                (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha);
      }
      lck.unlock();
    }
  }

  int nf = pModel->n_factors;
  for (int i = 0; i < xLen; i++) {
    ftrl_model_unit &mu = *params[i];
    int fi = std::get<0>(x[i]);
    for (int f = 0; f < nf; f++) {
      std::unique_lock<std::mutex> lck(mu.mtx);
      double &vif = mu.vi[fi * nf + f];
      double &v_nif = mu.v_ni[fi * nf + f];
      double &v_zif = mu.v_zi[fi * nf + f];
      if (fabs(v_zif) <= w_l1) {
        vif = 0.0;
      } else {
        vif = -1 * (v_zif - utils::sgn(v_zif) * w_l1) /
              (w_l2 + (w_beta + sqrt(v_nif)) / w_alpha);
      }
      lck.unlock();
    }
  }

  for (int i = 0; i < xLen; i++) {
    for (int j = i + 1; j < xLen; j++) {
      int fi1 = std::get<0>(x[i]);
      int fi2 = std::get<0>(x[j]);
      ftrl_model_unit &mu1 = *params[i];
      ftrl_model_unit &mu2 = *params[j];
      for (int f = 0; f < nf; f++) {
        std::unique_lock<std::mutex> lck1(mu1.mtx, std::defer_lock);
        std::unique_lock<std::mutex> lck2(mu2.mtx, std::defer_lock);
        std::lock(lck1, lck2);
        double &vif1 = mu1.vi[fi2 * nf + f];
        double &v_nif1 = mu1.v_ni[fi2 * nf + f];
        double &v_zif1 = mu1.v_zi[fi2 * nf + f];
        if (fabs(v_zif1) <= w_l1) {
          vif1 = 0.0;
        } else {
          vif1 = -1 * (v_zif1 - utils::sgn(v_zif1) * w_l1) /
                 (w_l2 + (w_beta + sqrt(v_nif1)) / w_alpha);
        }

        double &vif2 = mu2.vi[fi1 * nf + f];
        double &v_nif2 = mu2.v_ni[fi1 * nf + f];
        double &v_zif2 = mu2.v_zi[fi1 * nf + f];
        if (fabs(v_zif2) <= w_l1) {
          vif2 = 0.0;
        } else {
          vif2 = -1 * (v_zif2 - utils::sgn(v_zif2) * w_l1) /
                 (w_l2 + (w_beta + sqrt(v_nif2)) / w_alpha);
        }
        lck1.unlock();
        lck2.unlock();
      }
    }
  }

  double p = pModel->logit(x, true);
  double mult = y * (1 / (1 + exp(-p * y)) - 1);
  for (int i = 0; i <= xLen; i++) {
    if ((i < xLen && k1) || (i == xLen && k0)) {
      ftrl_model_unit &mu = *params[i];
      double xi = i < xLen ? std::get<2>(x[i]) : 1.0;
      std::unique_lock<std::mutex> lck(mu.mtx);
      double w_gi = mult * xi;
      double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
      mu.w_zi += w_gi - w_si * mu.wi;
      mu.w_ni += w_gi * w_gi;
      lck.unlock();
    }
  }

  for (int i = 0; i < xLen; i++) {
    ftrl_model_unit &mu = *params[i];
    const double &xi = std::get<2>(x[i]);
    int fi = std::get<0>(x[i]);
    for (int f = 0; f < nf; f++) {
      std::unique_lock<std::mutex> lck(mu.mtx);
      double &vif = mu.vi[fi * nf + f];
      double &v_nif = mu.v_ni[fi * nf + f];
      double &v_zif = mu.v_zi[fi * nf + f];
      double &s_vx = pModel->sum_vx[f];
      double v_gif = mult * (xi * s_vx - vif * xi * xi);
      double v_sif = 1 / w_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
      v_zif += v_gif - v_sif * vif;
      v_nif += v_gif * v_gif;
      lck.unlock();
    }
  }

  for (int i = 0; i < xLen; i++) {
    for (int j = i + 1; j < xLen; j++) {
      int fi1 = std::get<0>(x[i]);
      int fi2 = std::get<0>(x[j]);
      ftrl_model_unit &mu1 = *params[i];
      ftrl_model_unit &mu2 = *params[j];
      const double &xi = std::get<2>(x[i]) * std::get<2>(x[j]);
      for (int f = 0; f < nf; f++) {
        std::unique_lock<std::mutex> lck1(mu1.mtx, std::defer_lock);
        std::unique_lock<std::mutex> lck2(mu2.mtx, std::defer_lock);
        std::lock(lck1, lck2);
        double &vif1 = mu1.vi[fi2 * nf + f];
        double &v_nif1 = mu1.v_ni[fi2 * nf + f];
        double &v_zif1 = mu1.v_zi[fi2 * nf + f];
        double &vif2 = mu2.vi[fi1 * nf + f];
        double &v_nif2 = mu2.v_ni[fi1 * nf + f];
        double &v_zif2 = mu2.v_zi[fi1 * nf + f];

        double v_gif1 = mult * vif2 * xi;
        double v_sif1 = 1 / w_alpha * (sqrt(v_nif1 + v_gif1 * v_gif1) - sqrt(v_nif1));
        v_zif1 += v_gif1 - v_sif1 * vif1;
        v_nif1 += v_gif1 * v_gif1;
        double v_gif2 = mult * vif1 * xi;
        double v_sif2 = 1 / w_alpha * (sqrt(v_nif2 + v_gif2 * v_gif1) - sqrt(v_nif2));
        v_zif2 += v_gif2 - v_sif2 * vif2;
        v_nif2 += v_gif2 * v_gif2;
        lck1.unlock();
        lck2.unlock();
      }
    }
  }
}

}

#endif //FTRL_FFM_FTRL_TRAINER_H