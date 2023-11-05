#include "tank.hpp"


Tank::Tank(cv::Mat tank_model) : tank{tank_model} {
    w = tank.cols;
    h = tank.rows;
    tank_center = cv::Point(w / 2, h / 2);
    orientation = Keys::KEY_UP;

    back = cv::Mat::zeros(720, 1280, CV_8UC3);
    back_center = cv::Point(back.cols / 2, back.rows / 2);
}


void Tank::spawn(cv::Mat frame, cv::Point spawn_pt) {
    tank.copyTo(frame(cv::Rect(spawn_pt.x - w / 2, spawn_pt.y - h / 2, w, h)));
}


bool Tank::checkMoveUp(cv::Point spawn_pt, int new_w, int new_h) {
    return spawn_pt.y > new_h / 2 + step;
}


bool Tank::checkMoveDown(cv::Point spawn_pt, int new_w, int new_h) {
    return spawn_pt.y < back.rows - new_h / 2 - step;
}


bool Tank::checkMoveLeft(cv::Point spawn_pt, int new_w, int new_h) {
    return spawn_pt.x > new_w / 2 + step;
}


bool Tank::checkMoveRight(cv::Point spawn_pt, int new_w, int new_h) {
    return spawn_pt.x < back.cols - new_w / 2 - step;
}


bool Tank::checkSpawnPosition(cv::Point spawn_pt, int new_w, int new_h) {
    if (checkMoveUp(spawn_pt, new_w, new_h) && checkMoveUp(spawn_pt, new_w, new_h) &&
                                           checkMoveUp(spawn_pt, new_w, new_h) && checkMoveUp(spawn_pt, new_w, new_h)) {
        return true;
    }
    return false;
}


bool Tank::rotate(cv::Point spawn_pt, int r) {
    bool ret_flag = false;

    if (orientation == Keys::KEY_UP) {
        if (r == Keys::KEY_DOWN && checkMoveDown(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_180);
            orientation = Keys::KEY_DOWN;
            ret_flag = true;
        }
        if (r == Keys::KEY_LEFT && checkMoveLeft(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_90_COUNTERCLOCKWISE);
            orientation = Keys::KEY_LEFT;
            ret_flag = true;
        }
        if (r == Keys::KEY_RIGHT && checkMoveRight(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_90_CLOCKWISE);
            orientation = Keys::KEY_RIGHT;
            ret_flag = true;
        }
    }

    if (orientation == Keys::KEY_DOWN) {
        if (r == Keys::KEY_UP && checkMoveUp(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_180);
            orientation = Keys::KEY_UP;
            ret_flag = true;
        }
        if (r == Keys::KEY_LEFT && checkMoveLeft(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_90_CLOCKWISE);
            orientation = Keys::KEY_LEFT;
            ret_flag = true;
        }
        if (r == Keys::KEY_RIGHT && checkMoveRight(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_90_COUNTERCLOCKWISE);
            orientation = Keys::KEY_RIGHT;
            ret_flag = true;
        }
    }

    if (orientation == Keys::KEY_LEFT) {
        if (r == Keys::KEY_DOWN && checkMoveDown(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_90_COUNTERCLOCKWISE);
            orientation = Keys::KEY_DOWN;
            ret_flag = true;
        }
        if (r == Keys::KEY_UP && checkMoveUp(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_90_CLOCKWISE);
            orientation = Keys::KEY_UP;
            ret_flag = true;
        }
        if (r == Keys::KEY_RIGHT && checkMoveRight(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_180);
            orientation = Keys::KEY_RIGHT;
            ret_flag = true;
        }
    }

    if (orientation == Keys::KEY_RIGHT) {
        if (r == Keys::KEY_DOWN && checkMoveDown(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_90_CLOCKWISE);
            orientation = Keys::KEY_DOWN;
            ret_flag = true;
        }
        if (r == Keys::KEY_LEFT && checkMoveLeft(spawn_pt, h, w)) {
            cv::rotate(tank, tank, cv::ROTATE_180);
            orientation = Keys::KEY_LEFT;
            ret_flag = true;
        }
        if (r == Keys::KEY_UP && checkMoveUp(spawn_pt, w, h)) {
            cv::rotate(tank, tank, cv::ROTATE_90_COUNTERCLOCKWISE);
            orientation = Keys::KEY_UP;
            ret_flag = true;
        }
    }

    return ret_flag;
}


void Tank::createShell(cv::Point spawn_pt) {
    cv::Point shell_pt;
    if (orientation == Keys::KEY_UP) {
        shell_pt.x = spawn_pt.x;
        shell_pt.y = spawn_pt.y - h/2 - 6;
        // std::cout << spawn_pt.x << " " << spawn_pt.y << " " << shell_pt.x << " " << shell_pt.y << std::endl;
    }
    if (orientation == Keys::KEY_DOWN) {
        shell_pt.x = spawn_pt.x;
        shell_pt.y = spawn_pt.y + h/2 + 6;
    }
    if (orientation == Keys::KEY_LEFT) {
        shell_pt.x = spawn_pt.x - w/2 - 6;
        shell_pt.y = spawn_pt.y;
    }
    if (orientation == Keys::KEY_RIGHT) {
        shell_pt.x = spawn_pt.x + w/2 + 6;
        shell_pt.y = spawn_pt.y;
    }
    Shell::createShell(shell_pt, orientation);
}


void Tank::play() {
    cv::Mat frame;
    back.copyTo(frame);
    cv::Point spawn_pt = back_center;
    spawn(frame, spawn_pt);

    while (true) {

        back.copyTo(frame);
        spawn(frame, spawn_pt);
        Shell::draw(frame);
        imshow("Tanks", frame);
        int key = cv::waitKey(20);

        if (key == Keys::KEY_ESC) {
            return;
        }
        if (key == Keys::KEY_SPACE) {
            createShell(spawn_pt);
        }
        if (key == Keys::KEY_UP) {
            if (orientation == Keys::KEY_UP && checkMoveUp(spawn_pt, w, h)) {
                spawn_pt.y -= step;
            }
            rotate(spawn_pt, Keys::KEY_UP);
        }
        if (key == Keys::KEY_DOWN) {
            if (orientation == Keys::KEY_DOWN && checkMoveDown(spawn_pt, w, h)) {
                spawn_pt.y += step;
            }
            rotate(spawn_pt, Keys::KEY_DOWN);
        }
        if (key == Keys::KEY_LEFT) {
            if (orientation == Keys::KEY_LEFT && checkMoveLeft(spawn_pt, w, h)) {
                spawn_pt.x -= step;
            }
            rotate(spawn_pt, Keys::KEY_LEFT);
        }
        if (key == Keys::KEY_RIGHT) {
            if (orientation == Keys::KEY_RIGHT && checkMoveRight(spawn_pt, w, h)) {
                spawn_pt.x += step;
            }
            rotate(spawn_pt, Keys::KEY_RIGHT);
        }
    }
}
