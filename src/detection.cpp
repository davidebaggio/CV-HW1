// created by Davide Baggio 2122547

#include "detection.hpp"

string get_filename(string path)
{
	string filename = path.substr(path.find_last_of("/") + 1);
	return filename.substr(0, filename.find_last_of("-"));
}

bool is_yellow(Vec3b pixel)
{
	return (pixel[0] < 30 && pixel[1] > 90 && pixel[2] > 90);
}

bool is_dark(Vec3b pixel)
{
	return (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40);
}

bool is_white(Vec3b pixel)
{
	return (pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180);
}

bool is_red(Vec3b pixel)
{
	return (pixel[0] < 40 && pixel[1] < 40 && pixel[2] > 180);
}

bool is_blue(Vec3b pixel)
{
	return (pixel[0] > 180 && pixel[1] < 40 && pixel[2] < 40);
}

float intersection_over_union(Rect rect1, Rect rect2)
{
	Rect intersection = rect1 & rect2;
	float intersection_area = intersection.area();
	float union_area = rect1.area() + rect2.area() - intersection_area;
	return intersection_area / union_area;
}

void display_performances()
{
	vector<String> tested_annotations;
	glob("output/*.txt", tested_annotations, false);

	vector<String> true_labels, true_label_s, true_label_m, true_label_d;
	glob(base + sugar + label_path + "*.txt", true_label_s, false);
	glob(base + mustard + label_path + "*.txt", true_label_m, false);
	glob(base + drill + label_path + "*.txt", true_label_d, false);

	true_labels.insert(true_labels.end(), true_label_s.begin(), true_label_s.end());
	true_labels.insert(true_labels.end(), true_label_m.begin(), true_label_m.end());
	true_labels.insert(true_labels.end(), true_label_d.begin(), true_label_d.end());

	int obj_detected_num = 0;
	int obj_total = 0;
	float iou_total = 0;
	for (size_t i = 0; i < true_labels.size(); i++)
	{
		for (size_t j = 0; j < tested_annotations.size(); j++)
		{
			if (get_filename(true_labels[i]) != get_filename(tested_annotations[j]))
				continue;

			ifstream label_file(true_labels[i]);
			ifstream tested_file(tested_annotations[j]);

			if (!label_file.is_open() || !tested_file.is_open())
			{
				cerr << "[ERROR]: Could not open label or tested file." << endl;
				return;
			}
			string label_line, tested_line;
			while (getline(label_file, label_line))
			{
				while (getline(tested_file, tested_line))
				{
					string label_class = label_line.substr(0, label_line.find(" "));
					string tested_class = tested_line.substr(0, tested_line.find(" "));
					if (label_class == tested_class)
					{
						obj_total++;
						string label_box = label_line.substr(label_line.find(" ") + 1);
						string tested_box = tested_line.substr(tested_line.find(" ") + 1);

						vector<string> label_box_values, tested_box_values;
						string label_box_value, tested_box_value;
						stringstream label_box_ss(label_box);
						stringstream tested_box_ss(tested_box);

						while (getline(label_box_ss, label_box_value, ' '))
						{
							label_box_values.push_back(label_box_value);
						}
						while (getline(tested_box_ss, tested_box_value, ' '))
						{
							tested_box_values.push_back(tested_box_value);
						}

						Rect label_rect = Rect(stoi(label_box_values[0]), stoi(label_box_values[1]), stoi(label_box_values[2]) - stoi(label_box_values[0]), stoi(label_box_values[3]) - stoi(label_box_values[1]));
						Rect tested_rect = Rect(stoi(tested_box_values[0]), stoi(tested_box_values[1]), stoi(tested_box_values[2]) - stoi(tested_box_values[0]), stoi(tested_box_values[3]) - stoi(tested_box_values[1]));

						float IoU = intersection_over_union(label_rect, tested_rect);
						iou_total += IoU;
						cout << get_filename(true_labels[i]) << " - " << label_class << " -> IoU: " << IoU << endl;
						if (IoU > 0.5)
						{
							obj_detected_num++;
						}
					}
				}
				tested_file.clear();
				tested_file.seekg(0, ios::beg);
			}
			label_file.close();
			tested_file.close();
		}
	}

	cout << "Total objects detected: " << obj_detected_num << "/" << obj_total << endl;
	cout << "Average IoU: " << iou_total / static_cast<float>(obj_total) << endl;
}