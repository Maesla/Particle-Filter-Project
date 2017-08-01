/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  int M = 100;
  this->num_particles = M;

  // TODO: Create normal distributions for y and psi
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_psi(theta, std[2]);
  default_random_engine gen;
  for(int i = 0; i < M; i++)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_psi(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

  is_initialized = true;

}

void ParticleFilter::printParticles()
{
  for(int i = 0; i < num_particles; i++)
  {
    cout << particles[i].id << ": " << particles[i].x << endl;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_psi(0, std_pos[2]);

  int M = num_particles;
  for(int i = 0; i < M; i++)
  {
    updateMotionModel(particles[i], delta_t, velocity, yaw_rate);

    //Adding Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_psi(gen);

  }

  //printParticles();

}

void ParticleFilter::updateMotionModel(Particle &p, double delta_t,double velocity, double yaw_rate)
{
  bool isYawRateZero = fabs(yaw_rate) < 0.01;
  double delta_x;
  double delta_y;

  double theta = p.theta;
  if (isYawRateZero)
  {
    delta_x = velocity*delta_t*cos(theta);
    delta_y = velocity*delta_t*sin(theta);
  }
  else
  {
    delta_x = (velocity/yaw_rate)*(sin(theta + yaw_rate*delta_t)-sin(theta));
    delta_y = (velocity/yaw_rate)*(cos(theta)-cos(theta+yaw_rate*delta_t));
  }

  p.x += delta_x;
  p.y += delta_y;
  p.theta += yaw_rate*delta_t;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
  int M = num_particles;
  for(int i = 0; i < M; i++)
  {
    std::vector<LandmarkObs> predictedObservations;
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      Particle p = particles[i];
      Map::single_landmark_s landmark = map_landmarks.landmark_list[j];

      float theta = p.theta;
      float localx = landmark.x_f - p.x;
      float localy = landmark.y_f - p.y;
      float dist = sqrt(localx + localy);

      if (dist < sensor_range)
      {
        float rotatedLocalx = localx*cos(theta) - localy*sin(theta);
        float rotatedLocaly = localx*sin(theta) + localy*cos(theta);
        LandmarkObs predictedObservation;
        predictedObservation.id = landmark.id_i;
        predictedObservation.x = rotatedLocalx;
        predictedObservation.y = rotatedLocaly;
        predictedObservations.push_back(predictedObservation);

      }

    }
  }
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

