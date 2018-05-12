/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "map.h"

using namespace std;

// 根据初始位置，初始化粒子滤波器
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // 设置总共粒子的数量为1000
  num_particles = 1000;

  // 在原GPS值上，加上高斯噪声
  normal_distribution<> norm_x(x, std[0]);
  normal_distribution<> norm_y(y, std[1]);
  normal_distribution<> norm_theta(theta, std[2]);
  default_random_engine generator;

  // 把生成的粒子放到列表
  for (int i = 0; i < num_particles; i++) {
    double x_norm = norm_x(generator);
    double y_norm = norm_y(generator);
    double theta_norm = norm_theta(generator);
    struct Particle particle = { i, x_norm, y_norm, theta_norm, 1 };
    particles.push_back(particle);
  }

  // 设置所有粒子已经初始化完成
  is_initialized = true;
  cout << "初始化粒子滤波器完成" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // 根据上一时刻的状态，预测下一时刻的状态，状态中不包括速度
  for (int i = 0; i < particles.size(); i++) {
    Particle particle = particles[i];
    double p_x = particle.x;
    double p_y = particle.y;
    double p_theta = particle.theta;
    // 根据公式预测下一状态
    double px_p, py_p;
    if (fabs(yaw_rate) > 0.001) {
      px_p = p_x + velocity / yaw_rate * (sin(p_theta + yaw_rate * delta_t) - sin(p_theta));
      py_p = p_y + velocity / yaw_rate * (cos(p_theta) - cos(p_theta + yaw_rate * delta_t));
    } else {
      px_p = p_x + velocity * delta_t * cos(p_theta);
      py_p = p_y + velocity * delta_t * sin(p_theta);
    }
    double theta_p = p_theta + yaw_rate * delta_t;

    // 在原GPS值上，加上高斯噪声
    default_random_engine generator;
    normal_distribution<> norm_x(px_p, std_pos[0]);
    normal_distribution<> norm_y(py_p, std_pos[1]);
    //normal_distribution<> norm_theta(theta_p, std_pos[2]);

    // 更新粒子的状态值
    px_p = norm_x(generator);
    py_p = norm_y(generator);
    //theta_p = norm_theta(theta_p);
    particle.x = px_p;
    particle.y = py_p;
    particle.theta = theta_p;
  }
  cout << "预测完成" << endl;
}

/**
 * 数据联合（预测点和测量点）
 **/
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  // 遍历每一个测量点，把它归属于最近的地标点
  for (int i = 0; i < observations.size(); i++) {
    double min_dist = 0;
    int min_id = 0;
    // 是否已经对min_dist和min_id初始化
    bool inited = false;
    for (int j = 0; j < predicted.size(); j++) {
      int id = predicted[j].id;
      double result = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (!inited || result < min_dist) {
        min_dist = result;
        min_id = id;
        inited = true;
      }
    }
    observations[i].id = min_id;
  }
}

/**
 * 使用多元高斯更新权重
 **/
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &obs,
                                   const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < particles.size(); i++) {
    // 获取当前粒子坐标和角度值
    Particle particle = particles[i];
    double p_x = particle.x;
    double p_y = particle.y;
    double p_theta = particle.theta;

    // 根据地图上在当前粒子sensor_range范围内的地标点，计算到当前粒子的伪距离，其实就是预测的地标点，封装到集合中
    std::vector<LandmarkObs> predicted;
    for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
      Map::single_landmark_s mark = map_landmarks.landmark_list[i];
      double result = dist(p_x, p_y, mark.x_f, mark.y_f);
      if (result < sensor_range) {
        double pseudo_range_x = mark.x_f - p_x;
        double pseudo_range_y = mark.y_f - p_y;
        struct LandmarkObs markObs = { mark.id_i, pseudo_range_x, pseudo_range_y };
        predicted.push_back(markObs);
      }
    }

    // 因为在c++中不存在引用的引用，所以把observations复制一份新的
    std::vector<LandmarkObs> observations(obs);

    // 对于每一个观察值，计算地图上最近的地标是哪一个
    dataAssociation(predicted, observations);

    // 更新当前粒子的associations、sense_x、sense_y
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (int i = 0; i < observations.size(); i++) {
      associations.push_back(observations[i].id);
      sense_x.push_back(observations[i].x);
      sense_y.push_back(observations[i].y);
    }
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    // 对每一个观察值，转换到地图坐标，使用多元高斯计算权重
    double total_weight = 1;
    for (int j = 0; j < observations.size(); j++) {
      // 转换坐标，包括平移和旋转
      LandmarkObs obs = observations[j];
      double x_obs = obs.x;
      double y_obs = obs.y;
      double x_map = p_x + (cos(p_theta) * x_obs) - (sin(p_theta) * y_obs);
      double y_map = p_y + (sin(p_theta) * x_obs) + (cos(p_theta) * y_obs);
      // 获取该观察值最近的地标点
      Map::single_landmark_s mark = map_landmarks.landmark_list[observations[j].id];
      // 使用多元高斯分布，计算当前观察值的概率
      double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent = pow((x_map - mark.x_f), 2) / (2 * std_landmark[0] * std_landmark[0])
          + pow((y_map - mark.y_f), 2) / (2 * std_landmark[1] * std_landmark[1]);
      total_weight = total_weight * gauss_norm * exp(-exponent);
    }
    particle.weight = total_weight;
  }
  cout << "权重更新完成" << endl;
}

/**
 *根据权重，重采样
 **/
void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // 把粒子滤波的权重建立一个权重数组
  int particles_size = particles.size();
  vector<double> weights;
  for (int i = 0; i < particles_size; i++) {
    Particle particle = particles[i];
    weights.push_back(particle.weight);
  }

  // 根据新建的权重数组，进行重采样,
  // discrete_distribution会根据其中每个粒权重定义为wi/S，即第i个粒子的权重除以所有n个权重之和
  discrete_distribution<> d(weights.begin(), weights.end());
  default_random_engine generator;
  std::vector<Particle> new_particles;
  for (int i = 0; i < particles_size; i++) {
    int index = d(generator);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
  cout << "重采样完成" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space

  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space

  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space

  return s;
}
