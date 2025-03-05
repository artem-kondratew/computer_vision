#include "shell.hpp"


Shell::Shell(cv::Point shell_pt, int orientation) {
    if (orientation == Keys::KEY_UP) {
        is_x = false;
        step = -MAIN_STEP;
    }
    if (orientation == Keys::KEY_DOWN) {
        is_x = false;
        step = MAIN_STEP;
    }
    if (orientation == Keys::KEY_LEFT) {
        is_x = true;
        step = -MAIN_STEP;
    }
    if (orientation == Keys::KEY_RIGHT) {
        is_x = true;
        step = MAIN_STEP;
    }
    pt = shell_pt;
}


void Shell::createShell(cv::Point shell_pt, int orientation) {
    auto new_shell = new Shell(shell_pt, orientation);
    list.push_back(new_shell);
}


void Shell::increaseSteps(cv::Mat frame) {
    auto it = list.begin();
    while (it != list.end()) {
        auto shell = *it;
        if (shell->is_x) {
            shell->pt.x += shell->step;
            if (shell->pt.x > frame.cols || shell->pt.x < 0) {
                it = list.erase(it);
            }
        }
        else {
            shell->pt.y += shell->step;
            if (shell->pt.y > frame.rows || shell->pt.y < 0) {
                it = list.erase(it);
            }
        }
        it++;
    }
}


void Shell::draw(cv::Mat frame) {
    for (auto shell: list) {
        cv::Rect rec(cv::Point{shell->pt.x - 6, shell->pt.y - 6}, cv::Point{shell->pt.x + 6, shell->pt.y + 6});
        cv::rectangle(frame, rec, {0, 0, 255}, -1);
    }
    increaseSteps(frame);
}
