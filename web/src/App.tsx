import ClusterMap from './ClusterMap.tsx'

const GITHUB_URL = 'https://github.com/ch345/nowthennow'

function isAboutPath(pathname: string) {
  return pathname.replace(/\/+$/, '').endsWith('/about')
}

export default function App() {
  if (isAboutPath(window.location.pathname)) {
    return (
      <main className="intro">
        <a href={GITHUB_URL}>github.com/ch345/nowthennow</a>
      </main>
    )
  }

  return (
    <main className="intro">
      <h1>now then now</h1>
      <p>all our complex lives all at once</p>
      <ClusterMap />
    </main>
  )
}
