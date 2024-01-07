from simple_image_download import simple_image_download as simp


def download_images():
    my_downloader = simp.simple_image_download()
    keywords = ["speed30 sign",
                "stop sign",
                "traffic light",
                "crosswalk sign"]
    for kw in keywords:
        my_downloader.download(kw, limit=40)
    print("Finished downloading images.")


if __name__ == '__main__':
    download_images()
