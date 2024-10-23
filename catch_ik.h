#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define M_PI 3.14159265358979323846

#include<stdbool.h>
#include <iostream> 
#include "mujoco/mujoco.h"
#include "GLFW/glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;
int body,dof;
const double link2_length = 2.05;
double claw_x = 0.0;
double claw_y = 0.0;
double claw_z = 0.0;

bool press = false;
bool forward_ = false;
bool backward = false;
bool leftward = false;
bool rightward = false;
bool up = false;
bool down = false;
bool grip = false;
bool turn_left = false;
bool turn_right = false;


mjtNum point[3] = { 0 }; // ĩ��ִ�����Ĳο���

void init_model()
{
    char path[] = "./model/";
    char xmlfile[] = "myproject.xml";
    char xmlpath[100];

    strncpy(xmlpath, path, sizeof(xmlpath));
    strncat(xmlpath, xmlfile, sizeof(xmlpath) - strlen(xmlpath) - 1);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    m = mj_loadXML(xmlpath, 0, error, 1000);

    if (!m)
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);
    mj_step(m, d);
	mj_forward(m, d);

	d->ctrl[7] = 100;
	d->ctrl[9] = 100;
	d->ctrl[11] = 100;

	body = mj_name2id(m, mjOBJ_BODY, "link6");
	printf("body = %d\n", body);

	int total_joints = m->njnt;
	// ��ȡ���ɹؽڵ�����
	int free_joints = 0;
	for (int i = 0; i < total_joints; ++i) {
		if (m->jnt_type[i] == mjJNT_FREE) {
			free_joints++;
		}
	}
	printf("Total joints: %d\n", total_joints);
	printf("Free joints: %d\n", free_joints);
	// ���ɶ�����
	dof = total_joints+ 5 * free_joints;
	printf("dof = %d\n", dof);

	claw_x = d->xpos[3 * body];
	claw_y = d->xpos[3 * body + 1];
	claw_z = d->xpos[3 * body + 2];

}

void my_keyboard(GLFWwindow* window)
{
    // ����Ƿ�סĳ����
    forward_ = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    leftward = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    rightward = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
    up = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
    down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
    grip = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
    turn_left = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
    turn_right = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
    press = (forward_ || backward || leftward || rightward || up || down);

}


void my_gotoxyz(const mjModel* m, mjData* d, double x, double y, double z)
{
    double angle = atan2(y, x);
    double distance = sqrt(x * x + y * y);

    // �����е�ֹؽ�ת���Ƕ�
    double alpha = 2 * asin(distance / link2_length);

    // ���е��ת������Ӧλ��ע�ⲻ��ͻ��
    d->ctrl[0] = 2 * angle;
    d->ctrl[1] = alpha;
    d->ctrl[2] = 0.9 * alpha;
}

// ����ת����ת��Ϊŷ���ǣ�roll, pitch, yaw�ֱ��ʾ��x, y, z�����ת�Ƕ�
void rotationMatrixToEulerAngles(const mjtNum* R, double& roll, double& pitch, double& yaw) {
	if (R[2 * 3 + 0] < 1) {
		if (R[2 * 3 + 0] > -1) {
			pitch = asin(-R[2 * 3 + 0]);
			roll = atan2(R[2 * 3 + 1], R[2 * 3 + 2]);
			yaw = atan2(R[1 * 3 + 0], R[0 * 3 + 0]);
		}
		else {
			pitch = M_PI / 2;
			roll = -atan2(-R[1 * 3 + 2], R[1 * 3 + 1]);
			yaw = 0;
		}
	}
	else {
		pitch = -M_PI / 2;
		roll = atan2(-R[1 * 3 + 2], R[1 * 3 + 1]);
		yaw = 0;
	}
}

MatrixXd JacobianMatrix(const mjModel* m, mjData* d, int body) {
	std::vector<mjtNum> jacp(3 * dof, 0);
	std::vector<mjtNum> jacr(3 * dof, 0);
	mjtNum point[3] = { 0 }; // ĩ��ִ�����Ĳο���
	mj_jac(m, d, jacp.data(), jacr.data(), point, body);
	MatrixXd J(6, 6);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 6; ++j) {
			J(i, j) = jacp[i * dof + j];
			J(i + 3, j) = jacr[i * dof + j];
		}
	}
	return J;
}

/*****************************************************
 �ṩ�ƶ��ı仯����delta_pos������ؽڽǶȱ仯
 ����������delta_pos��һ��6ά������(1,0,0,0,0,0)
 ���һ��6ά��������ʾÿ���ؽڵĽǶȱ仯(*,*,*,*,*,*)
*****************************************************/
VectorXd move_ik(const mjModel* m, mjData* d, VectorXd delta_pos, MatrixXd J)
{
	// �����ſɱȾ����α��
	MatrixXd J_pseudo_inv = J.transpose() * (J * J.transpose()).inverse();

	// ����ؽڱ仯�ǶȦ�
	VectorXd delta_theta (6);
	delta_theta = J_pseudo_inv * delta_pos;
	for (int i = 0; i < 6; i++)
	{
		printf("delta_theta[%d] = %f\n", i, delta_theta(i));
	}
	printf("*****************************************************\n");
	return delta_theta;
}

/************************************************
 ���˶�ѧ,����Ŀ��λ�ã�����ؽڽǶ�
 target_pos��һ��3ά����,��ʾ��������ϵ��Ŀ��λ��
 ʹ��ţ��-����ɭ������,��ÿ����Ŀ�귽����һС����ʵʱ��̬����
 ÿ��ѭ���õ�һ��6ά��������ʾÿ���ؽڵĽǶ�(*,*,*,*,*,*)��Ȼ��ֱ�ӵ����Ƕ�
 ��һ��ѭ��ʱ�ٻ�ȡ�µ��ſɱȾ����ٴε����Ƕ�
************************************************/
void inverse_kinematics(const mjModel* m, mjData* d, VectorXd& target_pos, MatrixXd J) 
{

	for (int iter = 0; iter < 100; iter++) {
		mj_forward(m, d);

		// ����λ�����
		mjtNum* current_pos = d->xpos + 3 * body;
		// ������̬��ͨ����ת����ת��Ϊŷ���ǣ�
		mjtNum* current_rot = d->xmat + 9 * body;
		double roll, pitch, yaw;
		rotationMatrixToEulerAngles(current_rot, roll, pitch, yaw);

		VectorXd pos_error(6);
		pos_error << target_pos[0] - current_pos[0], target_pos[1] - current_pos[1], target_pos[2] - current_pos[2], -roll, -pitch, -yaw;
		double temp = 0.0;
		for (int i = 0; i < 6; i++) {
			temp += pos_error(i) * pos_error(i);
		}
		if (temp < 0.001) {
			break;
		}

		// ����ؽڱ仯�ǶȦȣ���������ģ��Ϊ0.1
		VectorXd delta_theta = VectorXd::Zero(6);
		delta_theta = move_ik(m, d, pos_error, J);
		delta_theta = delta_theta.normalized() * 0.1;
		

		// ���¹ؽڽǶ�
		for (int i = 0; i < 6; ++i) {
			d->ctrl[i] += delta_theta(i);
		}
	}
}


void mycontroller(const mjModel* m, mjData* d)
{
	double k = 0.01;
	// ����������
	my_keyboard(glfwGetCurrentContext());
	VectorXd delta_pos=VectorXd::Zero(6);
	// ���ƻ�е���˶�
	if (forward_)
	{
		claw_x = d->xpos[3 * body]+ k;
		delta_pos(1) += -k;
	}
	if (backward)
	{
		claw_x = d->xpos[3 * body]- k;
		delta_pos(1) += k;
	}
	if (leftward)
	{
		claw_y = d->xpos[3 * body+1]+ k;
		delta_pos(0) += k;
	}
	if (rightward)
	{
		claw_y = d->xpos[3 * body + 1]- k;
		delta_pos(0) += -k;
	}
	if (up)
	{
		claw_z = d->xpos[3 * body + 2]+ k;
		delta_pos(2) += k;
	}
	if (down)
	{
		claw_z = d->xpos[3 * body + 2]- k;
		delta_pos(2) += -k;
	}

	if (grip) {
		d->ctrl[6] = -100;
		d->ctrl[8] = -100;
		d->ctrl[10] = -100;
	}
	else {
		d->ctrl[6] = 100;
		d->ctrl[8] = 100;
		d->ctrl[10] = 100;
	}

	if (turn_left) {
		d->ctrl[5] += k;
	}
	else if (turn_right) {
		d->ctrl[5] -= k;
	}
	if (press) 
	{
		MatrixXd J = JacobianMatrix(m, d, body);
		VectorXd delta_theta = move_ik(m, d, delta_pos,J);
		for (int i =0; i<6; i++)
		{
			d->ctrl[i] += delta_theta(i);
			printf("delta_theta[%d] = %f\n", i, delta_theta(i));
		}
		printf("*****************************************************\n");
	}


	delta_pos = VectorXd::Zero(6);
}
