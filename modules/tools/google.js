const googlethis = require("googlethis");

function timeoutPromise(time) {
  return new Promise(function (resolve) {
    setTimeout(function () {
      resolve(Date.now());
    }, time);
  });
}

function doSomethingAsync() {
  return timeoutPromise(1000);
}

async function asyncSearch() {
  try {
    const args = process.argv.slice(2);
    const options = {
      page: 0,
      safe: false,
      parse_ads: false,
      additional_params: {
        // add additional parameters here, see https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters and https://www.seoquake.com/blog/google-search-param/
      },
    };
    const response = await googlethis.search(args[0], options);
    // console.log(response);
    console.log(JSON.stringify({
      results: response.results,
      featured: response.featured_snippet,
    }));
  } catch (error) {
    console.log(error);
    console.log("Failed to get results from Google. Do not try using Google again.");
  }
}

asyncSearch();
